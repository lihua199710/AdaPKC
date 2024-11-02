"""Main script to test a pretrained model"""
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adapkc.utils.paths import Paths
from adapkc.utils.functions import count_params
from adapkc.learners.tester import Tester
from adapkc.models import TMVANet, MVNet, MVANet, PKCIn, PKCOn, AdaPKC_Theta, AdaPKC_Xi, AdaPKC_Xi_Faster, PKCIn_AdaSample
from adapkc.loaders.dataset import Carrada
from adapkc.loaders.dataloaders import SequenceCarradaDataset
from adapkc.utils.distributed_utils import init_distributed_mode


def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file of the model to test.', default='config.json')
    parser.add_argument('--dist-url', default='env://', help='Url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action='store_true')
    parser.add_argument("--model-path", dest="model_path", help="Path of trained model")
    args = parser.parse_args()

    init_distributed_mode(args)
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)
    device = torch.device(cfg['device'])
    cfg['distributed'] = args.distributed

    # path of trained model
    model_path = args.model_path

    if cfg['model'] == 'mvnet':
        model = MVNet(n_classes=cfg['nb_classes'],
                    n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'mvanet':
        model = MVANet(n_classes=cfg['nb_classes'],
                     n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'pkcon':
        model = PKCOn(n_classes=cfg['nb_classes'],
                    n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'pkcin':
        model = AdaPKC_Xi(n_classes=cfg['nb_classes'], 
                    n_frames=cfg['nb_input_channels'],
                    threshold=1.0)
    elif cfg['model'] == 'adapkc_theta':
        model = AdaPKC_Theta(n_classes=cfg['nb_classes'], 
                    n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'adapkc_xi':
        model = AdaPKC_Xi(n_classes=cfg['nb_classes'], 
                    n_frames=cfg['nb_input_channels'],
                    threshold=0.0)
    elif cfg['model'] == 'adapkc_xi_faster':
        model = AdaPKC_Xi_Faster(n_classes=cfg['nb_classes'], 
                    n_frames=cfg['nb_input_channels'],
                    threshold=0.0)
    elif cfg['model'] == 'adapkc_finetune':
        model = AdaPKC_Xi(n_classes=cfg['nb_classes'], 
                    n_frames=cfg['nb_input_channels'],
                    threshold=cfg['conf_thre'])
    elif cfg['model'] == 'pkcin_adasample':
        model = PKCIn_AdaSample(n_classes=cfg['nb_classes'], 
                    n_frames=cfg['nb_input_channels'])
    else:
        model = TMVANet(n_classes=cfg['nb_classes'],
                      n_frames=cfg['nb_input_channels'])
    
    print('Number of trainable parameters in the model: %s' % str(count_params(model)))

    saved_model = torch.load(model_path, map_location=torch.device('cpu'))
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.load_state_dict(saved_model['state_dict'])
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    tester = Tester(cfg)
    data = Carrada()
    test = data.get('Test')
    testset = SequenceCarradaDataset(test)
    seq_testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    tester.set_annot_type(cfg['annot_type'])
    if cfg['model'] == 'mvnet':
        test_metrics = tester.predict(model, seq_testloader, get_quali=False, add_temp=False)
    else:
        test_metrics = tester.predict(model, seq_testloader, get_quali=False, add_temp=True)

    print('Test losses: '
        'RD={}, RA={}'.format(test_metrics['range_doppler']['loss'],
                            test_metrics['range_angle']['loss']))
    print('Test Prec: '
        'RD={}, RA={}'.format(test_metrics['range_doppler']['prec'],
                                test_metrics['range_angle']['prec']))
    print('Test mIoU: '
        'RD={}, RA={}'.format(test_metrics['range_doppler']['miou'],
                                test_metrics['range_angle']['miou']))
    print('Test Dice: '
        'RD={}, RA={}'.format(test_metrics['range_doppler']['dice'],
                                test_metrics['range_angle']['dice']))

if __name__ == '__main__':
    test_model()
