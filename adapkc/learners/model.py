"""Class to train a PyTorch model"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from adapkc.loaders.dataloaders import CarradaDataset
from adapkc.learners.tester import Tester
from adapkc.utils.functions import normalize, define_loss, get_transformations, warmup_lr_scheduler
from adapkc.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer
from adapkc.utils.distributed_utils import get_rank, reduce_value
import random
import time


class Model(nn.Module):
    """Class to train a model

    PARAMETERS
    ----------
    net: PyTorch Model
        Network to train
    data: dict
        Parameters and configurations for training
    """

    def __init__(self, net, data):
        super().__init__()
        self.net = net
        self.cfg = data['cfg']
        self.paths = data['paths']
        self.dataloaders = data['dataloaders']
        self.resume_path = self.cfg['resume']
        self.model_name = self.cfg['model']
        self.process_signal = self.cfg['process_signal']
        self.dataset_type = self.cfg['dataset_type']
        self.annot_type = self.cfg['annot_type']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.batch_size = self.cfg['batch_size']
        self.nb_epochs = self.cfg['nb_epochs']
        self.thre = self.cfg['thre']
        self.thre_step = self.cfg['thre_step']
        self.lr = self.cfg['lr']
        self.lr_step = self.cfg['lr_step']
        self.schedular_type = self.cfg['schedular']
        self.T_max = self.cfg['Tmax']
        self.warmup = self.cfg['warmup']
        self.warmup_factor = self.cfg['warmup_factor']
        self.warmup_iters = self.cfg['warmup_iters']
        self.loss_step = self.cfg['loss_step']
        self.val_step = self.cfg['val_step']
        self.viz_step = self.cfg['viz_step']
        self.torch_seed = self.cfg['torch_seed']
        self.numpy_seed = self.cfg['numpy_seed']
        self.nb_classes = self.cfg['nb_classes']
        self.custom_loss = self.cfg['custom_loss']
        self.comments = self.cfg['comments']
        self.n_frames = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.is_shuffled = self.cfg['shuffle']
        self.device = self.cfg['device']
        self.distributed = self.cfg['distributed']
        self.finetune = self.cfg['finetune']
        self.num_workers = self.cfg['num_workers']
        self.rank = get_rank()
        if self.rank == 0:
            self.writer = SummaryWriter(self.paths['writer'])
            self.visualizer = TensorboardMultiLossVisualizer(self.writer)
            self.tester = Tester(self.cfg, self.visualizer)
        else:
            self.tester = Tester(self.cfg)
        self.results = dict()

    def train(self, add_temp=False):
        """
        Method to train a network

        PARAMETERS
        ----------
        add_temp: boolean
            Add a temporal dimension during training?
            Considering the input as a sequence.
            Default: False
        """
        if self.rank == 0:
            self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        self._set_seeds()
        rd_criterion = define_loss('range_doppler', self.custom_loss, self.device)
        ra_criterion = define_loss('range_angle', self.custom_loss, self.device)
        nb_losses = len(rd_criterion)
        running_losses = list()
        rd_running_losses = list()
        rd_running_global_losses = [list(), list()]
        ra_running_losses = list()
        ra_running_global_losses = [list(), list()]
        coherence_running_losses = list()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # select learning rate schedular of exp or cos.
        if self.schedular_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max-1, eta_min=1e-7)
        else:
            raise KeyError("we only implement schedular of exp and cos")
        start_epoch = 0
        iteration = 0
        # lt @20230522
        flag_save = False
        name_result = []
        best_val_dop = 0
        best_test_dop = 0
        
        # resume training
        if self.resume_path:
            checkpoint = torch.load(self.resume_path)
            start_epoch = checkpoint['epoch']
            self.net.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_val_dop = checkpoint['best_val_dop']
            best_test_dop = checkpoint['best_test_dop']
            print("=> Loaded checkpoint at epoch {} w/ best dice in valuation: {}, best dice in testing: {})".format(
                checkpoint['epoch'], best_val_dop, best_test_dop))

        # warmup strategy, lt@20230416
        if self.warmup:
            warm_scheduler = warmup_lr_scheduler(optimizer, self.warmup_iters, self.warmup_factor)
        for epoch in range(start_epoch, self.nb_epochs):
            # adjust confidence threshold linearly lt@20230617
            if self.thre and epoch % self.thre_step == 0 and epoch > 0:
                self.net.module.thre_step()

            if self.warmup and iteration < self.warmup_iters:
                warm_scheduler.step()
            
            if self.schedular_type == 'exp':
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            else:
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()
            
            for seq_idx, sequence_data in enumerate(train_loader):
                seq_name, seq = sequence_data
                path_to_frames = os.path.join(self.paths['carrada'], seq_name[0])
                carrada_dataset = CarradaDataset(seq,
                                                self.dataset_type,
                                                self.annot_type,
                                                path_to_frames,
                                                self.process_signal,
                                                self.n_frames,
                                                transformations,
                                                add_temp)
                
                if self.distributed:
                    sampler_train = torch.utils.data.distributed.DistributedSampler(carrada_dataset)
                    sampler_train.set_epoch(epoch)
                else:
                    sampler_train = torch.utils.data.RandomSampler(carrada_dataset)
                train_batch_sampler = torch.utils.data.BatchSampler(sampler_train, self.batch_size, drop_last=True)
                frame_dataloader = DataLoader(carrada_dataset,
                                              batch_sampler=train_batch_sampler,
                                              num_workers=self.num_workers)
                for _, frame in enumerate(frame_dataloader):
                    rd_data = frame['rd_matrix'].to(self.device).float()
                    ra_data = frame['ra_matrix'].to(self.device).float()
                    ad_data = frame['ad_matrix'].to(self.device).float()
                    rd_mask = frame['rd_mask'].to(self.device).float()
                    ra_mask = frame['ra_mask'].to(self.device).float()
                    rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type)
                    ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                    # zlw@20220325
                    if self.model_name != 'mvnet':
                        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type)
                    optimizer.zero_grad()
                    # zlw@20220325
                    if self.model_name != 'mvnet':
                        rd_outputs, ra_outputs = self.net(rd_data, ra_data, ad_data)
                    else:
                        rd_outputs, ra_outputs = self.net(rd_data, ra_data)
                    rd_outputs = rd_outputs.to(self.device)
                    ra_outputs = ra_outputs.to(self.device)
                    if nb_losses < 3:
                        # Case without the CoL
                        rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                     for c in rd_criterion]
                        rd_loss = torch.mean(torch.stack(rd_losses))
                        ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                     for c in ra_criterion]
                        ra_loss = torch.mean(torch.stack(ra_losses))
                        loss = torch.mean(rd_loss + ra_loss)
                        # aggregate loss from all devices
                        rd_losses_reduced = reduce_value(rd_losses)
                        rd_loss_reduced = reduce_value(rd_loss)
                        ra_losses_reduced = reduce_value(ra_losses)
                        ra_loss_reduced = reduce_value(ra_loss)
                        loss_reduced = reduce_value(loss)
                    else:
                        # Case with the CoL
                        # Select the wCE and wSDice
                        rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                     for c in rd_criterion[:2]]
                        rd_loss = torch.mean(torch.stack(rd_losses))
                        ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                     for c in ra_criterion[:2]]
                        ra_loss = torch.mean(torch.stack(ra_losses))
                        # Coherence loss
                        coherence_loss = rd_criterion[2](rd_outputs, ra_outputs)
                        loss = torch.mean(rd_loss + ra_loss + coherence_loss)
                        # aggregate loss from all devices
                        rd_losses_reduced = reduce_value(rd_losses)
                        rd_loss_reduced = reduce_value(rd_loss)
                        ra_losses_reduced = reduce_value(ra_losses)
                        ra_loss_reduced = reduce_value(ra_loss)
                        coherence_loss_reduced = reduce_value(coherence_loss)
                        loss_reduced = reduce_value(loss)

                    loss.backward()
                    optimizer.step()
                    # print only in main process
                    if self.rank == 0:
                        running_losses.append(loss_reduced.data.cpu().numpy()[()])
                        rd_running_losses.append(rd_loss_reduced.data.cpu().numpy()[()])
                        rd_running_global_losses[0].append(rd_losses_reduced[0].data.cpu().numpy()[()])
                        rd_running_global_losses[1].append(rd_losses_reduced[1].data.cpu().numpy()[()])
                        ra_running_losses.append(ra_loss_reduced.data.cpu().numpy()[()])
                        ra_running_global_losses[0].append(ra_losses_reduced[0].data.cpu().numpy()[()])
                        ra_running_global_losses[1].append(ra_losses_reduced[1].data.cpu().numpy()[()])
                        if nb_losses > 2:
                            coherence_running_losses.append(coherence_loss_reduced.data.cpu().numpy()[()])

                        if iteration % self.loss_step == 0:
                            train_loss = np.mean(running_losses)
                            rd_train_loss = np.mean(rd_running_losses)
                            rd_train_losses = [np.mean(sub_loss) for sub_loss in rd_running_global_losses]
                            ra_train_loss = np.mean(ra_running_losses)
                            ra_train_losses = [np.mean(sub_loss) for sub_loss in ra_running_global_losses]
                            # zlw@20220302
                            print('[{}][Epoch {}/{}, iter {}]: '
                                'Train loss {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                        epoch+1,
                                                        self.nb_epochs,
                                                        iteration,
                                                        train_loss))
                            print('[Epoch {}/{}, iter {}]: '
                                'Train losses: RD={}, RA={}'.format(epoch+1,
                                                                    self.nb_epochs,
                                                                    iteration,
                                                                    rd_train_loss,
                                                                    ra_train_loss))
                            if nb_losses > 2:
                                coherence_train_loss = np.mean(coherence_running_losses)
                                # zlw@20220302
                                print('[Epoch {}/{}, iter {}]: '
                                    'Train Coherence loss {}'.format(epoch+1,
                                                                    self.nb_epochs,
                                                                    iteration,
                                                                    coherence_train_loss))
                            if nb_losses > 2:
                                self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                        rd_train_losses, ra_train_loss,
                                                                        ra_train_losses, iteration,
                                                                        coherence_train_loss)
                            else:
                                self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                        rd_train_losses, ra_train_loss,
                                                                        ra_train_losses, iteration)
                            running_losses = list()
                            rd_running_losses = list()
                            ra_running_losses = list()
                            # zlw@20220107 get_lr() --> get_last_lr()
                            self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)

                    if iteration % self.val_step == 0 and iteration > 0:
                        if iteration % self.viz_step == 0 and iteration > 0:
                            val_metrics = self.tester.predict(self.net, val_loader, iteration,
                                                              add_temp=add_temp)
                        else:
                            val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp)

                        if self.rank == 0:
                            self.visualizer.update_multi_val_metrics(val_metrics, iteration)
                            print('[Epoch {}/{}] Val losses: '
                                'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['loss'],
                                                        val_metrics['range_angle']['loss'],))
                            if nb_losses > 2:
                                print('[Epoch {}/{}] Val Coherence Loss: '
                                'CoL={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['coherence_loss']))
                            print('[Epoch {}/{}] Val Pixel Prec: '
                                'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['prec'],
                                                        val_metrics['range_angle']['prec']))
                            # zlw@20220227
                            print('[Epoch {}/{}] Val mIoU: '
                                'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['miou'],
                                                        val_metrics['range_angle']['miou']))
                            print('[Epoch {}/{}] Val Dice: '
                                'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['dice'],
                                                        val_metrics['range_angle']['dice']))
                            

                        # lt @20230522
                        if val_metrics['range_doppler']['dice'] > best_val_dop and iteration > 0:
                            best_val_dop = val_metrics['range_doppler']['dice']
                            flag_save = True
                            name_result.append('val_doppler')
                        
                        # saving results w/ best dice zlw@20220704
                        test_metrics = self.tester.predict(self.net, test_loader,
                                                               add_temp=add_temp)
                        if self.rank == 0:
                            self.visualizer.update_multi_test_metrics(test_metrics, iteration)
                            print('[Epoch {}/{}] Test losses: '
                                'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['loss'],
                                                        test_metrics['range_angle']['loss']))
                            print('[Epoch {}/{}] Test Prec: '
                                'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['prec'],
                                                        test_metrics['range_angle']['prec']))
                            # zlw@20220227
                            print('[Epoch {}/{}] Test mIoU: '
                                'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['miou'],
                                                        test_metrics['range_angle']['miou']))
                            print('[Epoch {}/{}] Test Dice: '
                                'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['dice'],
                                                        test_metrics['range_angle']['dice']))

                            self.results['epoch'] = epoch
                            self.results['rd_train_loss'] = rd_train_loss.item()
                            self.results['ra_train_loss'] = ra_train_loss.item()
                            self.results['train_loss'] = train_loss.item()
                            self.results['val_metrics'] = val_metrics
                            self.results['test_metrics'] = test_metrics
                            if nb_losses > 3:
                                self.results['coherence_train_loss'] = coherence_train_loss.item()
                            
                        if test_metrics['range_doppler']['dice'] > best_test_dop and iteration > 0:
                            best_test_dop = test_metrics['range_doppler']['dice']
                            flag_save = True
                            name_result.append('test_doppler')
                        
                        if flag_save:
                            state = {
                                'epoch': epoch + 1,
                                'arch': str(self.net.module),
                                'state_dict': self.net.module.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'best_val_dop': best_val_dop,
                                'best_test_dop': best_test_dop,
                            }
                            self._save_results(name_result, state)
                            flag_save = False
                            name_result = []
                        
                        self.net.train()  # Train mode after evaluation process
                    iteration += 1
        if self.rank == 0:
            self.writer.close()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0., 1.)
                nn.init.constant_(m.bias, 0.)

    def _save_results(self, names, state):
        for name in names:
            if self.rank == 0:
                results_path = self.paths['results'] / (name + '_' + 'results.json')
                with open(results_path, "w") as fp:
                    json.dump(self.results, fp)
                model_path = self.paths['results'] / (name + '_' + 'model.pt')
                torch.save(state, model_path)

    def _set_seeds(self):
        torch.cuda.manual_seed(self.torch_seed+get_rank())
        torch.manual_seed(self.torch_seed+get_rank())
        np.random.seed(self.numpy_seed+get_rank())
        random.seed(self.numpy_seed+get_rank())
