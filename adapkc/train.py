"""Main script to train a model"""
import argparse
import json
from adapkc.utils.functions import count_params
from adapkc.learners.initializer import Initializer
from adapkc.learners.model import Model
from adapkc.models import TMVANet, MVNet, MVANet, TransRad, PKCIn, PKCOn, AdaPKC_Theta, AdaPKC_Xi, AdaPKC_Xi_Faster, PKCIn_AdaSample
import os
import torch.nn as nn
import torch
import numpy as np
import random
from adapkc.utils.distributed_utils import init_distributed_mode, get_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.', default='config.json')
    parser.add_argument('--dist-url', default='env://', help='Url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action='store_true')
    parser.add_argument("--finetune", default=False, help="The path of pretrained model in the finetuning stage")
    args = parser.parse_args()
    print(args)

    init_distributed_mode(args)

    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)
    device = torch.device(cfg['device'])
    cfg['distributed'] = args.distributed
    cfg['finetune'] = args.finetune

    init = Initializer(cfg)
    data = init.get_data()
    if cfg['model'] == 'mvnet':
        net = MVNet(n_classes=data['cfg']['nb_classes'],
                    n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'mvanet':
        net = MVANet(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'transradar':
        net = TransRad(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcon':
        net = PKCOn(n_classes=data['cfg']['nb_classes'],
                    n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcin':
        net = AdaPKC_Xi(n_classes=data['cfg']['nb_classes'], 
                    n_frames=data['cfg']['nb_input_channels'],
                    threshold=1.0)
    elif cfg['model'] == 'adapkc_theta':
        net = AdaPKC_Theta(n_classes=data['cfg']['nb_classes'], 
                    n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'adapkc_xi':
        net = AdaPKC_Xi(n_classes=data['cfg']['nb_classes'], 
                    n_frames=data['cfg']['nb_input_channels'],
                    threshold=0.0)
    elif cfg['model'] == 'adapkc_xi_faster':
        net = AdaPKC_Xi_Faster(n_classes=data['cfg']['nb_classes'], 
                    n_frames=data['cfg']['nb_input_channels'],
                    threshold=0.0)
    elif cfg['model'] == 'adapkc_finetune':
        net = AdaPKC_Xi(n_classes=data['cfg']['nb_classes'], 
                    n_frames=data['cfg']['nb_input_channels'],
                    threshold=data['cfg']['conf_thre'])
    elif cfg['model'] == 'pkcin_adasample':
        net = PKCIn_AdaSample(n_classes=data['cfg']['nb_classes'], 
                    n_frames=data['cfg']['nb_input_channels'])
    else:
        net = TMVANet(n_classes=data['cfg']['nb_classes'],
                      n_frames=data['cfg']['nb_input_channels'])

    print('Number of trainable parameters in the model: %s' % str(count_params(net)))

    torch.manual_seed(cfg['torch_seed'])
    net.apply(_init_weights)
    
    if args.distributed and args.sync_bn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if args.finetune:
        saved_model = torch.load(args.finetune, map_location=torch.device('cpu'))
        net.load_state_dict(saved_model['state_dict'])

    net.to(device)
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    
    if cfg['model'] in ['mvnet']:
        Model(net, data).train(add_temp=False)
    else:
        Model(net, data).train(add_temp=True)

def _init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.uniform_(m.weight, 0., 1.)
        nn.init.constant_(m.bias, 0.)

if __name__ == '__main__':
    main()
