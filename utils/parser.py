import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)   
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--dataset_root', type = str, default='/data/dhegde1/data/3D/', help = 'experiment name')
    parser.add_argument('--train_dataset', type = str, default='shapenet55', help = 'experiment name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')

    parser.add_argument('--text_prompt', type = str, default='This is a ', help = 'test freq')

    parser.add_argument('--VL', type = str, default='SLIP', help = 'vision-language model')
    parser.add_argument('--out_dir', type = str, default='experiments_v3', help = 'vision-language model')
    parser.add_argument('--slip_model', type = str, default='/data/dhegde1/code/CLASP_pb/Point-BERT/models/SLIP/model_zoo/slip_base_100ep.pt', help = 'vision-language model')
    parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                        help='hidden dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-emb-dim', default=256, type=int,
                        help='output embed dim of SimCLR mlp projection head')
    parser.add_argument('--slip_model_name', default='SLIP_VITB16', type=str)

    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--dvae', 
        action='store_true', 
        default=False, 
        help = 'pointcloud recon')

    parser.add_argument(
        '--scene', 
        action='store_true', 
        default=False, 
        help = 'pointcloud recon')
    parser.add_argument(
        '--real_caption', 
        action='store_true', 
        default=False, 
        help = 'pointcloud recon')
    parser.add_argument(
        '--zshot', 
        action='store_true', 
        default=False, 
        help = 'pointcloud recon')
    parser.add_argument(
        '--pretrain', 
        action='store_true', 
        default=False, 
        help = 'pretrain')
    parser.add_argument(
        '--finetune', 
        action='store_true', 
        default=False, 
        help = 'finetune')
    parser.add_argument(
        '--finetune_image', 
        action='store_true', 
        default=False, 
        help = 'finetune')
    parser.add_argument(
        '--barlow', 
        action='store_true', 
        default=False, 
        help = 'use barlow twins')

    parser.add_argument(
        '--random_sample', 
        action='store_true', 
        default=False, 
        help = 'use barlow twins')
    parser.add_argument(
        '--real', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--visual_prompting', 
        action='store_true', 
        default=False, 
        help = 'use VPT for vision encoder')
    parser.add_argument(
        '--original', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--no_pb', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model', 
        action='store_true', 
        default=False, 
        help = 'finetune modelnet with pretrained weight')
    parser.add_argument(
        '--image', 
        action='store_true', 
        default=False, 
        help = 'point-image loss')
    parser.add_argument(
        '--depth', 
        action='store_true', 
        default=False, 
        help = 'point-depth image loss')
    parser.add_argument(
        '--cls', 
        action='store_true', 
        default=False, 
        help = 'point-image loss')
    parser.add_argument(
        '--text', 
        action='store_true', 
        default=False, 
        help = 'point-text loss')
    parser.add_argument(
        '--clip', 
        action='store_true', 
        default=False, 
        help = 'image-text loss')
    parser.add_argument(
        '--scratch_model', 
        action='store_true', 
        default=False, 
        help = 'training modelnet from scratch')
    parser.add_argument(
        '--label_smoothing', 
        action='store_true', 
        default=False, 
        help = 'use label smoothing loss trick')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')        
    parser.add_argument(
        '--way', type=int, default=-1)
    parser.add_argument(
        '--shot', type=int, default=-1)
    parser.add_argument(
        '--fold', type=int, default=-1)

    parser.add_argument(
        '--npoints', type=int, default=8192)

    parser.add_argument(
        '--per_samples', type=float, default=-1)

    parser.add_argument(
        '--aug_prob', type=float, default=0.3)
    
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    # if args.resume and args.start_ckpts is not None:
    #     raise ValueError(
    #         '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    # if args.finetune_model and args.ckpts is None:
    #     raise ValueError(
    #         'ckpts shouldnt be None while finetune_model mode')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode
    args.experiment_path = os.path.join(args.out_dir, Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join(args.out_dir, Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

