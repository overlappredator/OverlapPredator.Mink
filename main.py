import os, torch, time, shutil, json,glob, argparse, shutil
import numpy as np
from easydict import EasyDict as edict

from datasets.dataloader import get_datasets, collate_pair_fn
from lib.utils import setup_seed, load_config
from lib.tester import get_trainer
from lib.loss import MetricLoss
from models import load_model

from torch import optim
from torch import nn
setup_seed(0)


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    config['snapshot_dir'] = 'snapshot/%s' % config['exp_dir']
    config['tboard_dir'] = 'snapshot/%s/tensorboard' % config['exp_dir']
    config['save_dir'] = 'snapshot/%s/checkpoints' % config['exp_dir']
    config = edict(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
    
    # backup the files
    os.system(f'cp -r models {config.snapshot_dir}')
    os.system(f'cp -r datasets {config.snapshot_dir}')
    os.system(f'cp -r lib {config.snapshot_dir}')
    shutil.copy2('main.py',config.snapshot_dir)
    
    
    # model initialization
    Model = load_model(config.model)
    config.model = Model(config, D=3)
    print(config.model)

    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    
    # create learning rate scheduler
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )
    
    # create dataset and dataloader
    train_set, val_set, benchmark_set = get_datasets(config)
    config.train_loader = torch.utils.data.DataLoader(train_set, 
                                        batch_size=config.batch_size, 
                                        shuffle=True,
                                        num_workers=config.num_workers,
                                        collate_fn=collate_pair_fn,
                                        pin_memory=False,
                                        drop_last=False)
    config.val_loader = torch.utils.data.DataLoader(val_set, 
                                        batch_size=config.batch_size, 
                                        shuffle=True,
                                        num_workers=config.num_workers,
                                        collate_fn=collate_pair_fn,
                                        pin_memory=False,
                                        drop_last=False)
    config.test_loader = torch.utils.data.DataLoader(benchmark_set, 
                                        batch_size=config.batch_size, 
                                        shuffle=False,
                                        num_workers=config.num_workers,
                                        collate_fn=collate_pair_fn,
                                        pin_memory=False,
                                        drop_last=False)
    
    # create evaluation metrics
    config.desc_loss = MetricLoss(config)
    trainer = get_trainer(config)
    if(config.mode=='train'):
        trainer.train()
    elif(config.mode =='val'):
        trainer.eval()
    else:
        trainer.test()        