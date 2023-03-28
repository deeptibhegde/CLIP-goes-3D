import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms


from models.PointMLP import pointMLPProject as pointMLP


from tensorboardX import SummaryWriter

import math 
import os


from sklearn.cluster import KMeans

from models.dvae import Group, Group_Iterate


from PIL import Image

import torchvision.transforms as transforms

from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *

import models.SLIP.models as slip_models

import clip


train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    
    config = get_config(args, logger = logger)

    train_writer = None
    val_writer = None

    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    


    


    # DATALOADER #########################################################################

    logger = get_logger(args.log_name)

    start_epoch = 0
    # build dataset
    
    (_, MN40_dataloader,mn40_classes),(_, MN10_dataloader,mn10_classes),\
     (_, scan_dataloader,scan_classes)= builder.dataset_builder(args, config.dataset.mn40), \
                                        builder.dataset_builder(args, config.dataset.mn10), \
                                        builder.dataset_builder(args, config.dataset.scan) 


    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    
    


    ## MODEL DEF ########################################################################
    # build model
    # import pdb; pdb.set_trace()

    if config.model.NAME == 'PointMLP':
        base_model = pointMLP()
        base_model.load_model_from_ckpt(base_model,args.ckpts)
    else:
        base_model = builder.model_builder(config.model)
        base_ckpt = torch.load(args.ckpts)
        # base_ckpt = {k.replace("blocks.", ""): v for k, v in base_ckpt['base_model'].items()}
        # checkpoint = torch.load(args.ckpts)['base_model']
        base_model.load_state_dict(base_ckpt['base_model'])
        print("Loaded model from ",args.ckpts)
        # base_model.load_model_from_ckpt(args.ckpts)

    base_model.to(args.local_rank)


    
    if args.VL == 'CLIP':
        clip_model, preprocess = clip.load("RN50x16",device=args.local_rank,jit=False)

    elif args.VL == 'SLIP':
        # import pdb; pdb.set_trace()
        
        clip_model = getattr(slip_models, args.slip_model_name)(ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim).to(args.local_rank)
        pretrained_slip = torch.load(args.slip_model)
        temp = {}
        for key,value in pretrained_slip['state_dict'].items():
            k = key.replace("module.","")
            temp[k] = value

        
        clip_model.load_state_dict(temp,strict=False)
        clip_dict = torch.load(args.ckpts)['visual_clip_model']
        clip_model.visual.load_state_dict(clip_dict)
    clip_model.to(args.local_rank)
    # PREP FOR TESTING ################################################################
    
    f = open(os.path.join(args.dataset_root,"shapenet_render/shape_names.txt"))
    val_classes = f.readlines()
    for i in range(len(val_classes)):
        val_classes[i] = val_classes[i][:-1]

    texts_validation = []
    for c in val_classes:
        texts_validation.append(args.text_prompt + c)

    text_validation = clip.tokenize(texts_validation).to(args.local_rank)


    mn40_texts_validation = []
    for c in mn40_classes.keys():
        mn40_texts_validation.append(args.text_prompt + c)

    mn40_text_validation = clip.tokenize(mn40_texts_validation).to(args.local_rank)

    mn10_texts_validation = []
    for c in mn10_classes.keys():
        mn10_texts_validation.append(args.text_prompt + c)

    mn10_text_validation = clip.tokenize(mn10_texts_validation).to(args.local_rank)


    scan_texts_validation = []
    for c in scan_classes.keys():
        scan_texts_validation.append(args.text_prompt + c)

    scan_text_validation = clip.tokenize(scan_texts_validation).to(args.local_rank)

    ########################################################################################

    # TEST ###########################################################################

    if args.zhot:

        overall_acc_MN40, class_wise_acc_MN40, metrics_MN40 = validate_ZS(args,base_model,clip_model,MN40_dataloader,mn40_text_validation,mn40_classes,val_writer,start_epoch,logger,config)
        print_log("{{MODELNET40 overall accuracy: %.3f}}"%overall_acc_MN40,logger = logger) 
        print_log("{{MODELNET40 class-wise mean Accuracy: %.3f}}"%class_wise_acc_MN40,logger = logger)


        overall_acc_MN10, class_wise_acc_MN10, metrics_MN10 = validate_ZS(args,base_model,clip_model,MN10_dataloader,mn10_text_validation,mn10_classes,val_writer,start_epoch,logger,config)
        print_log("{{MODELNET10 overall accuracy: %.3f}}"%overall_acc_MN10,logger = logger) 
        print_log("{{MODELNET10 class-wise mean Accuracy: %.3f}}"%class_wise_acc_MN10,logger = logger)

        overall_acc_scan, class_wise_acc_scan, metrics_scan = validate_ZS(args,base_model,clip_model,scan_dataloader,scan_text_validation,scan_classes,val_writer,start_epoch,logger,config)
        print_log("{{ScanObjectNN overall accuracy: %.3f}}"%overall_acc_scan,logger = logger) 
        print_log("{{ScanObjectNN class-wise mean Accuracy: %.3f}}"%class_wise_acc_scan,logger = logger)
    
    else:
        cls_acc = validate(base_model,MN40_dataloader, val_writer, args, config)



    #############################################################################################

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img =  img.convert('RGB') 

    transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                lambda x: x.convert('RGB'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

    return transform(img)[:3]



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc



def validate_ZS(args,base_model,clip_model, test_dataloader,text_validation,val_classes_dict,val_writer,epoch,logger,config):
    overall_acc_sh = 0
    overall_count_sh = 0
    npoints = config.npoints
    # val_classes = val_classes.keys()
    val_classes = [key for key in val_classes_dict]
    # import pdb; pdb.set_trace()
    acc_sh = [0]*len(val_classes)
    acc_count_sh = [0]*len(val_classes)
    base_model.eval()


    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):

        
        points = data[0].to(args.local_rank)
        points = misc.fps(points,args.npoints)

        
        label = data[1]

        im = data[2].cuda()
       
        
        batch_size = points.shape[0]
                    

        with torch.no_grad():

            if base_model.__class__.__name__ == 'ModelProject':
                latent_point = base_model(points.permute(0,2,1).contiguous())
            else:
                ret, latent_point, _ = base_model(points)

            

            ## GET TEXT FEATURES OF CAPTIONS
            text_features = clip_model.encode_text(text_validation)

   
            # normalize features
            latent_point = (latent_point / latent_point.norm(dim=-1, keepdim=True))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  





            # compute similarity
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * latent_point @ text_features.t().float()
            logits_per_text = logits_per_image.t()

            probs = logits_per_image.softmax(dim=-1).cpu().numpy()




            for i in range(len(probs)):

                

                ind = np.argmax(probs[i])
                prediction = val_classes[ind]
                if prediction == val_classes[label[i]]:                        
                    acc_sh[label[i]] += 1
                    overall_acc_sh += 1
                overall_count_sh += 1
                acc_count_sh[label[i]] += 1 
    
    

    acc_sh = np.mean(np.array(acc_sh))
    overall_acc_sh /= overall_count_sh



    if args.distributed:
        torch.cuda.synchronize()





    return overall_acc_sh, acc_sh, Acc_Metric(overall_acc_sh)



def validate(base_model, test_dataloader, val_writer, args, config, logger = None):
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            if config.model.NAME == 'PointMLP' or config.model.NAME == 'PointConv':
                logits = base_model(points.permute(0,2,1).contiguous())
            else:
                logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.


    # Add testing results to TensorBoard
    return acc
    

if __name__ == '__main__':
    main()