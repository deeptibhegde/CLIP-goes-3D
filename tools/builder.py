import os, sys
# online package
import torch
# optimizer
import torch.optim as optim
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from datasets.dataset import ShapenetCG3D
from datasets.S3DISDataset import _S3DISDataset as S3DIS
# utils
from utils.logger import *
from utils.misc import *
from timm.scheduler import CosineLRScheduler

from torch.optim.lr_scheduler import LambdaLR

import math


def dataset_builder_clasp(args, config,train=True,real=False):
    # dataset = build_dataset_from_cfg(config._base_, config.others)
    resize = args.VL == 'SLIP'
    dataset = ShapenetCG3D(args, config,train,real)
    
    shuffle = True
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            # worker_init_fn = worker_init_fn, 
                                            shuffle=False,
                                            sampler = sampler)
        
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                # worker_init_fn=worker_init_fn
                                                )
        # import pdb; pdb.set_trace()
    return sampler, dataloader



def dataset_builder_S3DIS(args, config):
    # dataset = build_dataset_from_cfg(config._base_, config.others)
    resize = args.VL == 'SLIP'
    dataset = S3DIS(root=args.dataset_root, num_points=args.npoints              
            )
    shuffle = config.others.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            # worker_init_fn = worker_init_fn, 
                                            shuffle=False,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                # worker_init_fn=worker_init_fn
                                                )
    return sampler, dataloader, dataset.classes




def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    # import pdb; pdb.set_trace()

    # shuffle = config.others.subset == 'train'
    shuffle=True

    classes = dataset.classes

    if args.per_samples > 0 and config.others.subset == 'train':
        sub = list(range(0, len(dataset), int(100//(args.per_samples))))
        dataset = torch.utils.data.Subset(dataset, sub)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            # worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                # worker_init_fn=worker_init_fn
                                                )
        # import pdb; pdb.set_trace()
        
    return sampler, dataloader, classes

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_opti_sche(base_model, config,clip_model=None):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        if clip_model is not None:
            optimizer = optim.Adam(list(base_model.parameters()) + list(clip_model.parameters()), **opti_config.kwargs)
        else:

            optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.epochs,
                cycle_mul=1,
                lr_min=1e-6,
                cycle_decay=0.1,
                warmup_lr_init=sche_config.kwargs.min_lr,
                warmup_t=sche_config.kwargs.initial_epochs,
                cycle_limit=1,
                t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return optimizer, scheduler


def build_VPT_optimizer(params, config):
    opti_config = config.optimizer_clip


    if opti_config.kwargs.weight_decay > 0:
        if opti_config.type == 'adamw':

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in params
                            if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in params
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=opti_config.kwargs.base_lr,
            )
        else:
            _params = []
            for p in params:
                key, value = p
                # print(key)
                # if not value.requires_grad:
                #     continue
                lr = opti_config.kwargs.base_lr
                weight_decay = opti_config.kwargs.weight_decay
                if "last_layer.bias" in key:
                    # no regularization (weight decay) for last layer's bias
                    weight_decay = 0.0

                if opti_config.kwargs.bias_multiplier == 1.:
                    _params += [{
                        "params": [value],
                        "lr": lr,
                        "weight_decay": weight_decay
                    }]
                else:
                    if "bias" in key and "last_layer.bias" not in key:
                        # use updated lr for this param
                        lr_value = lr * opti_config.kwargs.bias_multiplier
                    else:
                        lr_value = lr

                    

                    _params += [{
                        "params": [value],
                        "lr": lr_value,
                        "weight_decay": weight_decay
                    }]

            if opti_config.type == 'adam':
                optimizer = optim.Adam(
                    _params,
                    lr=opti_config.kwargs.base_lr,
                    weight_decay=opti_config.kwargs.weight_decay,
                )
            else:
                optimizer = optim.SGD(
                    _params,
                    opti_config.kwargs.base_lr,
                    momentum=opti_config.kwargs.momentum,
                    weight_decay=opti_config.kwargs.weight_decay
                )
        return optimizer
    else:
        if opti_config.type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=opti_config.kwargs.base_lr
            )
        else:
            _params = []
            for p in params:
                key, value = p

                lr = opti_config.kwargs.base_lr

                if opti_config.kwargs.bias_multiplier == 1:
                    _params += [{
                        "params": [value],
                        "lr": lr,
                    }]
                else:
                    if "bias" in key and "last_layer.bias" not in key:
                        # use updated lr for this param
                        lr_value = lr * opti_config.kwargs.bias_multiplier
                    else:
                        lr_value = lr

                    
                    _params += [{
                        "params": [value],
                        "lr": lr_value,
                    }]
            optimizer = optim.SGD(
                _params,
                opti_config.kwargs.base_lr,
                momentum=opti_config.kwargs.momentum,
            )
        return optimizer



def build_VPT_scheduler(config,optimizer):
    sched_config = config.scheduler
    warmup = sched_config.kwargs.initial_epochs
    total_iters = sched_config.kwargs.epochs

    if sched_config.type == "cosine":
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup,
            t_total=total_iters
        )
    elif sched_config.type == "cosine_hardrestart":
        scheduler = WarmupCosineWithHardRestartsSchedule(
            optimizer,
            warmup_steps=warmup,
            t_total=total_iters
        )

    elif sched_config.type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "max",
            patience=5,
            verbose=True,
            factor=sched_config.LR_DECAY_FACTOR,
        )
    else:
        scheduler = None
    return scheduler


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps`.
        Decreases learning rate from 1. to 0. over remaining
            `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate
            follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(
            1, self.t_total - self.warmup_steps))
        return max(
            0.0,
            0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )


class WarmupCosineWithHardRestartsSchedule(LambdaLR):
    """ Linear warmup and then cosine cycles with hard restarts.
        Linearly increases learning rate from 0 to 1 over `warmup_steps`.
        If `cycles` (default=1.) is different from default, learning rate
            follows `cycles` times a cosine decaying learning rate
            (with hard restarts).
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(
            0.0,
            0.5 * (1. + math.cos(
                math.pi * ((float(self.cycles) * progress) % 1.0)))
        )


def resume_model(base_model, args, logger = None):
    if args.start_ckpts is not None:
        ckpt_path = args.start_ckpts
    else:
        ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    # base_ckpt = {k: v for k, v in base_ckpt.items() if "cls_head_finetune" not in k }
    base_model.load_state_dict(base_ckpt, strict = False)

    # parameter
    if args.start_ckpts is not None:
        start_epoch = 0
    else:       
        start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger = None):
    if args.start_ckpts is not None:
        pass 
    else:
        ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
        if not os.path.exists(ckpt_path):
            print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
            return 0, 0, 0
        print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
        # load state dict
        state_dict = torch.load(ckpt_path, map_location='cpu')
        # optimizer
        optimizer.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, optimizer, epoch, metrics=None, best_metrics=None, prefix=None, args=None, logger = None,clip_model=None):
    if args.local_rank == 0:
        if clip_model is not None:
            torch.save({
                        'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                        'visual_clip_model': clip_model.visual.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'epoch' : epoch,
                        'metrics' : metrics.state_dict() if metrics is not None else dict(),
                        'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                        }, os.path.join(args.experiment_path, prefix + '.pth'))
        else:
            torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict = False)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return 