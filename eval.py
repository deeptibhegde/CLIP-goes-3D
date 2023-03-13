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

from pytorch3d.loss import chamfer_distance

from models.PointMLP import pointMLPProject as pointMLP

from utils.contrastive_loss import ContrastiveLoss, BarlowTwins

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
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger = logger)

    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() 

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold



    


    # DATALOADER #########################################################################

    logger = get_logger(args.log_name)

    start_epoch = 0
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),\
    (_, MN40_dataloader,mn40_classes),(_, MN10_dataloader,mn10_classes),\
     (_, scan_dataloader,scan_classes)= builder.dataset_builder_clasp(args, config.dataset.train,real=args.real), \
                                                            builder.dataset_builder_clasp(args, config.dataset.train,train=False,real=args.real), \
                                                            builder.dataset_builder(args, config.dataset.mn40), \
                                                            builder.dataset_builder(args, config.dataset.mn10), \
                                                            builder.dataset_builder(args, config.dataset.scan) 


    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    
    


    ## MODEL DEF ########################################################################
    # build model
    if config.model.NAME == 'PointMLP':
        base_model = pointMLP()
        base_model.load_model_from_ckpt(base_model,args.ckpts)
    else:
        base_model = builder.model_builder(config.model)
        # base_model.load_model_from_ckpt(args.ckpts)
        checkpoint = torch.load(args.ckpts)['base_model']
        base_model.load_state_dict(checkpoint,strict=False)

    base_model.to(args.local_rank)

    if args.clip or args.text or args.image:
        if args.VL == 'CLIP':
            clip_model, preprocess = clip.load("RN50x16",device=args.local_rank,jit=False)

        elif args.VL == 'SLIP':
            # import pdb; pdb.set_trace()
            clip_model = getattr(slip_models, args.slip_model_name)(ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim).to(args.local_rank)
            if args.visual_prompting:
                for k, p in clip_model.named_parameters():
                    if "prompt" not in k:
                        p.requires_grad = False
                    else:
                        print(k)
            pretrained_slip = torch.load(args.slip_model)
            temp = {}
            for key,value in pretrained_slip['state_dict'].items():
                k = key.replace("module.","")
                temp[k] = value

            
            clip_model.load_state_dict(temp,strict=False)
            clip_dict = torch.load(args.ckpts)['visual_clip_model']
            clip_model.visual.load_state_dict(clip_dict)
    clip_model.to(args.local_rank)
    # PREP FOR VALIDATION ################################################################
    
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

    # VALIDATION ###########################################################################
    # path = "/data/dhegde1/data/3D/retrieval_test_images/"
    # prompt_list = ["This is a plant","This is a cabinet","A photo of a chair","This is a bed"]
    # for img_file in os.listdir(path):
    # for prompt in prompt_list:

    #     # pointcloud_retrieval(args,base_model,clip_model,MN40_dataloader,mn40_text_validation,mn40_classes,val_writer,os.path.join(path,img_file))
    #     pointcloud_retrieval(args,base_model,clip_model,MN40_dataloader,mn40_text_validation,mn40_classes,val_writer,None,prompt)

    # import pdb; pdb.set_trace()


    # scene_retrieval(args,base_model,clip_model,config,"This is a books","/data/dhegde1/data/3D/s3dis/trainval_fullarea/Area_5_hallway_1.npy")
    scene_retrieval(args,base_model,clip_model,config,"table","/data/dhegde1/data/3D/s3dis/trainval_fullarea/Area_1_office_1.npy")

    # overall_acc_sh, acc_sh, metric = validate(args,base_model,clip_model,test_dataloader,text_validation,val_classes,val_writer,start_epoch,logger,config)
    print_log("{{SHAPENET Validation overall accuracy: %.3f}}"%overall_acc_sh,logger = logger) 
    print_log("{{SHAPENET Validation class-wise mean Accuracy: %.3f}}"%acc_sh,logger = logger)

    overall_acc_im, acc_im, metric_im = validate_image(args,base_model,clip_model,test_dataloader,text_validation,val_classes,val_writer,start_epoch,logger)
    print_log("{{SHAPENET IMAGE Validation overall accuracy: %.3f}}"%overall_acc_im,logger = logger) 
    print_log("{{SHAPENET IMAGE Validation class-wise mean Accuracy: %.3f}}"%acc_im,logger = logger)

    overall_acc_MN40, class_wise_acc_MN40, metrics_MN40 = validate_ZS(args,base_model,clip_model,MN40_dataloader,mn40_text_validation,mn40_classes,val_writer,start_epoch,logger,config)
    print_log("{{MODELNET40 Validation overall accuracy: %.3f}}"%overall_acc_MN40,logger = logger) 
    print_log("{{MODELNET40 Validation class-wise mean Accuracy: %.3f}}"%class_wise_acc_MN40,logger = logger)


    overall_acc_MN10, class_wise_acc_MN10, metrics_MN10 = validate_ZS(args,base_model,clip_model,MN10_dataloader,mn10_text_validation,mn10_classes,val_writer,start_epoch,logger,config)
    print_log("{{MODELNET10 Validation overall accuracy: %.3f}}"%overall_acc_MN10,logger = logger) 
    print_log("{{MODELNET10 Validation class-wise mean Accuracy: %.3f}}"%class_wise_acc_MN10,logger = logger)

    overall_acc_scan, class_wise_acc_scan, metrics_scan = validate_ZS(args,base_model,clip_model,scan_dataloader,scan_text_validation,scan_classes,val_writer,start_epoch,logger,config)
    print_log("{{ScanObjectNN Validation overall accuracy: %.3f}}"%overall_acc_scan,logger = logger) 
    print_log("{{ScanObjectNN Validation class-wise mean Accuracy: %.3f}}"%class_wise_acc_scan,logger = logger)

    overall_acc_MN40_im, class_wise_acc_MN40_im, metrics_MN40_im = validate_MN40Image(args,base_model,clip_model,MN40_dataloader,mn40_text_validation,mn40_classes,val_writer,start_epoch,logger,config)
    print_log("{{MODELNET40 IMAGE overall accuracy: %.3f}}"%overall_acc_MN40_im,logger = logger) 
    print_log("{{MODELNET40 IMAGE class-wise mean Accuracy: %.3f}}"%overall_acc_MN40_im,logger = logger)


    # overall_acc_MN40_fuse, class_wise_acc_MN40_fuse, metrics_MN40_fuse = validate_MN40Fuse(args,base_model,clip_model,MN40_dataloader,mn40_text_validation,mn40_classes,val_writer,start_epoch,logger,config)
    # print_log("{{MODELNET40 FUSED overall accuracy: %.3f}}"%overall_acc_MN40_fuse,logger = logger) 
    # print_log("{{MODELNET40 FUSED class-wise mean Accuracy: %.3f}}"%overall_acc_MN40_fuse,logger = logger)

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

def pointcloud_retrieval(args,base_model,clip_model,test_dataloader,mn40_text_validation,mn40_classes,val_writer,image_file,text_prompt):
    

    sim_scores = []

    pcds = []
    object_name = text_prompt.split(" ")[-1]
    base_model.eval()
    # img = pil_loader(image_file).cuda()
    text_prompt_embed = clip.tokenize(text_prompt).cuda()
    with torch.no_grad():
        latent_text = clip_model.encode_text(text_prompt_embed)
        latent_text = (latent_text / latent_text.norm(dim=-1, keepdim=True))

    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):

        points = data[0].to(args.local_rank)

        ret, latent_point, _ = base_model(points)

        with torch.no_grad():
            # normalize features
            latent_point = (latent_point / latent_point.norm(dim=-1, keepdim=True))

            # latent_image = clip_model.encode_image(img.unsqueeze(0))

            sim_feat = latent_text @ latent_point.t()

            sorted,indices = torch.sort(sim_feat)

            sim_scores.append(sorted[:,-5:].detach().cpu().numpy())
            inds = indices[:,-5:]
            pcds.append(points[inds[0]].detach().cpu().numpy())

    sim_scores = torch.tensor(np.array(sim_scores)).view(-1).contiguous()
    pcds = torch.tensor(np.array(pcds)).view(-1,8192,3).contiguous()

    sorted,indices = torch.sort(sim_scores)

    out_pcds = pcds[indices[-5:]].numpy()

    # name = image_file.split('/')[-1]

    # np.save("/data/dhegde1/code/CLASP_pb/Point-BERT/experiments_v3/misc/text/%s"%name[:-4],out_pcds)
    np.save("/data/dhegde1/code/CLASP_pb/Point-BERT/experiments_v3/misc/text/%s"%object_name,out_pcds)




        



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def scene_retrieval(args,base_model,clip_model,config,text_query,filename):

    prompt_templates =[
        "There is a {category} in the scene.",
        "There is the {category} in the scene.",
        "a photo of a {category} in the scene.",
        "a photo of the {category} in the scene.",
        "a photo of one {category} in the scene.",
        "itap of a {category}.",
        "itap of my {category}.",
        "itap of the {category}.",
        "a photo of a {category}.",
        "a photo of my {category}.",
        "a photo of the {category}.",
        "a photo of one {category}.",
        "a photo of many {category}.",
        "a good photo of a {category}.",
        "a good photo of the {category}.",
        "a bad photo of a {category}.",
        "a bad photo of the {category}.",
        "a photo of a nice {category}.",
        "a photo of the nice {category}.",
        "a photo of a cool {category}.",
        "a photo of the cool {category}.",
        "a photo of a weird {category}.",
        "a photo of the weird {category}.",
        "a photo of a small {category}.",
        "a photo of the small {category}.",
        "a photo of a large {category}.",
        "a photo of the large {category}.",
        "a photo of a clean {category}.",
        "a photo of the clean {category}.",
        "a photo of a dirty {category}.",
        "a photo of the dirty {category}.",
        "a bright photo of a {category}.",
        "a bright photo of the {category}.",
        "a dark photo of a {category}.",
        "a dark photo of the {category}.",
        "a photo of a hard to see {category}.",
        "a photo of the hard to see {category}.",
        "a low resolution photo of a {category}.",
        "a low resolution photo of the {category}.",
        "a cropped photo of a {category}.",
        "a cropped photo of the {category}.",
        "a close-up photo of a {category}.",
        "a close-up photo of the {category}.",
        "a jpeg corrupted photo of a {category}.",
        "a jpeg corrupted photo of the {category}.",
        "a blurry photo of a {category}.",
        "a blurry photo of the {category}.",
        "a pixelated photo of a {category}.",
        "a pixelated photo of the {category}.",
        "a black and white photo of the {category}.",
        "a black and white photo of a {category}",
        "a plastic {category}.",
        "the plastic {category}.",
        "a toy {category}.",
        "the toy {category}.",
        "a plushie {category}.",
        "the plushie {category}.",
        "a cartoon {category}.",
        "the cartoon {category}.",
        "an embroidered {category}.",
        "the embroidered {category}.",
        "a painting of the {category}.",
        "a painting of a {category}.",
    ]



    scene = np.load(filename)

    #remove ceiling
    mask = np.uint8(scene[:,-1]) == 0
    scene = scene[~mask]

    #remove floor
    mask =  np.uint8(scene[:,-1]) == 1
    scene = scene[~mask]

    #remove wall
    mask =  np.uint8(scene[:,-1]) == 2
    scene = scene[~mask]

    #remove beam
    mask =  np.uint8(scene[:,-1]) == 3
    scene = scene[~mask]

    #remove column
    mask =  np.uint8(scene[:,-1]) == 4
    scene = scene[~mask]

    #remove window
    mask =  np.uint8(scene[:,-1]) == 5
    scene = scene[~mask]

    #remove clutter
    mask =  np.uint8(scene[:,-1]) == 12
    scene = scene[~mask]

    #remove board
    mask =  np.uint8(scene[:,-1]) == 11
    scene = scene[~mask]

    #remove door
    mask =  np.uint8(scene[:,-1]) == 6
    scene = scene[~mask]
        

    labels = np.uint8(scene[:,-1])


    #area6office6

    # mask = scene[:,1] < 1.2
    # scene = scene[mask]

    # mask = scene[:,0] > 3.5
    # scene = scene[mask]
    # mask = scene[:,1] > 2
    # scene = scene[mask]


    # mask = scene[:,1] > 7
    # scene = scene[mask]
    # mask = scene[:,0] < 3.5
    # scene = scene[mask]

    # mask = scene[:,0] > 3.7
    # scene = scene[mask]

    # mask = scene[:,1] > 0.5
    # scene = scene[mask]

    # mask = scene[:,1] < 7
    # scene = scene[mask]


    # mask = scene[:,0] < 3.7
    # scene = scene[mask]

    # mask = scene[:,1] > 0.5
    # scene = scene[mask]

    #office31 
    # mask = scene[:,0] < 3
    # scene = scene[mask]
    # mask = scene[:,1] > 9
    # scene = scene[mask]

    #office2_13
    # mask = scene[:,0] < 1.38
    # scene = scene[mask]

    # import pdb; pdb.set_trace()

    # scene = pc_normalize(scene[:,:3])
    # choice = np.random.choice(scene.shape[0],self.num_points)
    # choice = np.random.choice(scene.shape[0],20000)
    # scene = scene[choice]

    #downsample and normalize to unit sphere
    # scene = torch.tensor(pc_normalize(scene[:,:3])).cuda().float()

    num_groups = [5,10,20,30,40,50,100,200]

    scene = torch.tensor(scene[:,:3]).cuda().float()

    out_points = []

    center_shift = None

    sim_feat_all = []
    seg_points_all = []

    
    # for i in range(1):
    # num_group = 200 
    group_size = 8192


    # group_divider = Group_Iterate(num_group, group_size)
    # group_divider = Group(num_group, group_size)

    # # neighborhood, center = group_divider(scene.unsqueeze(0),center_shift)
    # neighborhood, center = group_divider(scene.unsqueeze(0))

    # # import pdb; pdb.set_trace()
    # seg_points = neighborhood + center.unsqueeze(2)
    # points = torch.tensor(pc_normalize(neighborhood[0].cpu().numpy())).cuda().float()

    k = 11

    segments = []
    segments_norm = []
    centers = []

    clustering = KMeans(n_clusters=k).fit(scene.cpu().detach().numpy())
    clabels = torch.tensor(clustering.labels_[:,np.newaxis])[:,0]

    num_points = 1024
    

    for clabel in clabels.unique():
        mask = clabels == clabel

        points = scene[mask]

        # choice = np.random.choice(points.shape[0],num_points)
        # points = points[choice]
        
        points = misc.fps(points.unsqueeze(0),num_points).squeeze()

        segments.append(points.unsqueeze(0))

        # import pdb; pdb.set_trace()


        points = torch.tensor(pc_normalize(points.cpu().numpy())).cuda().float().unsqueeze(0)

        
        # choice = np.random.choice(points.shape[0],4000)
        # points = points[choice].unsqueeze(0)
        # # points = misc.fps(points,4000).unsqueeze(0)

        segments_norm.append(points)


    segments_norm = torch.cat(segments_norm,dim=0)
    segments = torch.cat(segments,dim=0)

    sim_all = []
  
    with torch.no_grad():

        for prompt in prompt_templates[10]:
            prompt.replace("{category}",text_query)
            ret, latent_point, out = base_model(segments_norm)

            text_embedding = clip.tokenize(prompt).cuda()
            text_features = clip_model.encode_text(text_embedding)

            latent_point = (latent_point / latent_point.norm(dim=-1, keepdim=True))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
            sim = latent_point @ text_features.t()

            sim_all.append(sim)
        # sim_feat_all.append(sim)

    # seg_points_all.append(seg_points)
    
    sim = torch.cat(sim_all,dim=-1).sum(-1).unsqueeze(-1)
    # sim = torch.cat(sim_feat_all)
    # import pdb; pdb.set_trace()

    seg_points = segments   


        
    sim += abs(sim.min())
    sim /= sim.max()

    # sim = sim.unsqueeze(-1)

    sim = sim.repeat(1,k).detach().cpu().numpy()

    sim_mask = sim[:,0] > 0.85
    # sim_mask = sim > 0.45





    # center_shift = center[0][sim_mask]

    # center_shift[:,:2] += 0.4

    # center_shift = center_shift.unsqueeze(0)

    import matplotlib.pyplot as plt 

    colormap = plt.get_cmap('inferno')
    heatmap = colormap(sim)

    heatmap =  heatmap[:,0,:3] 
    heatmap = torch.tensor(heatmap).unsqueeze(1).repeat(1,num_points,1)[sim_mask]

    # import pdb; pdb.set_trace()


    heatmap_uniform = torch.tensor([[0.8,0,0]]).unsqueeze(1).repeat(seg_points[sim_mask].shape[0],seg_points[sim_mask].shape[1],1)

    # center = center[0].detach().cpu().numpy()

    background_points = torch.tensor(np.ones(seg_points[~sim_mask].shape)*0.5)


    out_points = torch.cat((seg_points[sim_mask],heatmap.cuda()),-1).view(-1,6).detach().cpu().numpy()
    # out_points = torch.cat((seg_points[sim_mask],heatmap_uniform.cuda()),-1).view(-1,6).detach().cpu().numpy()
    out_points_back = torch.cat((seg_points[~sim_mask],background_points.cuda()),-1).view(-1,6).detach().cpu().numpy()
    # out_points.append(torch.cat((seg_points[0][sim_mask],heatmap.cuda()),-1).view(-1,6).detach().cpu().numpy())
    out_points = np.vstack((out_points,out_points_back))
    # pcd_att_map = np.hstack((center,heatmap))

    # out_points = np.vstack(out_points)
    # import pdb; pdb.set_trace()

    name = filename.split('/')[-1][:-4]

    q = text_query.split(" ")[-1]

    np.save("/data/dhegde1/code/CLASP_pb/Point-BERT/experiments_v3/misc/%s_"%name + "%s_p3.npy"%q,out_points)

    import pdb; pdb.set_trace()







def validate(args,base_model,clip_model,test_dataloader,text_validation,val_classes,val_writer,epoch,logger,config):
    npoints = config.npoints
    overall_acc_sh = 0
    overall_count_sh = 0
    acc_sh = [0]*len(val_classes)
    acc_count_sh = [0]*len(val_classes)
    base_model.eval()
    # import pdb; pdb.set_trace()
    for idx, (taxonomy_ids, model_ids, points, (img,depth_im), caption, class_name,label) in enumerate(test_dataloader):
        
        # img = img.cuda().float()
        
        # fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
        # fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        # points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

        
        points = points.to(args.local_rank)
    
        points = misc.fps(points,args.npoints)



       
        
        batch_size = img.shape[0]
                    
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # latent_img = clip_model.encode_image(img).float()
            

            if base_model.__class__.__name__ == 'ModelProject':
                latent_point = base_model(points.permute(0,2,1).contiguous())
            else:
                import pdb; pdb.set_trace()
                ret, latent_point, out = base_model(points)

            ## GET TEXT FEATURES OF CAPTIONS
            text_features = clip_model.encode_text(text_validation)

                
            # normalize features
            latent_point = (latent_point / latent_point.norm(dim=-1, keepdim=True))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  


            # import pdb; pdb.set_trace()



            # compute similarity
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * latent_point @ text_features.t().float()
            logits_per_text = logits_per_image.t()

            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            # import pdb; pdb.set_trace()
                
            for i in range(len(probs)):

                ind = np.argmax(probs[i])
                prediction = val_classes[ind]
                if prediction == class_name[i]:                        
                    acc_sh[val_classes.index(class_name[i])] += 1
                    overall_acc_sh += 1
                overall_count_sh += 1
                acc_count_sh[val_classes.index(class_name[i])] += 1 

    for i in range(len(acc_sh)): acc_sh[i] /= acc_count_sh[i]

    acc_sh = np.mean(np.array(acc_sh))
    overall_acc_sh /= overall_count_sh

    # print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,overall_acc_sh), logger=logger)

    if args.distributed:
        torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', overall_acc_sh, epoch)


    return overall_acc_sh, acc_sh, Acc_Metric(overall_acc_sh)



def validate_image(args,base_model,clip_model,test_dataloader,text_validation,val_classes,val_writer,epoch,logger):
    overall_acc_sh = 0
    overall_count_sh = 0
    acc_sh = [0]*len(val_classes)
    acc_count_sh = [0]*len(val_classes)
    base_model.eval()
    for idx, (taxonomy_ids, model_ids, points, (img,depth_image), caption, class_name,label) in enumerate(test_dataloader):
        
        img = img.cuda().float()
        

        # points = points.cuda()


       
        
        batch_size = img.shape[0]
                    
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            latent_img = clip_model.encode_image(img).float()

            
            

            ## GET TEXT FEATURES OF CAPTIONS
            text_features = clip_model.encode_text(text_validation)

                
            # normalize features
            latent_img = (latent_img / latent_img.norm(dim=-1, keepdim=True))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  


            # import pdb; pdb.set_trace()



            # compute similarity
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * latent_img @ text_features.t().float()
            logits_per_text = logits_per_image.t()

            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            # import pdb; pdb.set_trace()
                
            for i in range(len(probs)):

                ind = np.argmax(probs[i])
                prediction = val_classes[ind]
                if prediction == class_name[i]:                        
                    acc_sh[val_classes.index(class_name[i])] += 1
                    overall_acc_sh += 1
                overall_count_sh += 1
                acc_count_sh[val_classes.index(class_name[i])] += 1 

    for i in range(len(acc_sh)): acc_sh[i] /= acc_count_sh[i]

    acc_sh = np.mean(np.array(acc_sh))
    overall_acc_sh /= overall_count_sh

    # print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,overall_acc_sh), logger=logger)

    if args.distributed:
        torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_Image', overall_acc_sh, epoch)


    return overall_acc_sh, acc_sh, Acc_Metric(overall_acc_sh)



def validate_(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, points, img, caption, class_name,label) in enumerate(test_dataloader):
            points = points.to(args.local_rank)
            label = label.to(args.local_rank)

            points = misc.fps(points, npoints)

            

            if base_model.__class__.__name__ == 'ModelProject':
                latent_point = base_model(points.permute(0,2,1).contiguous())
            else:
                logits, _, _ = base_model(points)

            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_cls', acc, epoch)

    return Acc_Metric(acc)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, points, img, caption, class_name,label) in enumerate(test_dataloader):
            points = points[0].to(args.local_rank)
            # points_raw = points
            # label = label[1].to(args.local_rank)
            # if npoints == 1024:
            #     point_all = 1200
            # elif npoints == 2048:
            #     point_all = 2400
            # elif npoints == 4096:
            #     point_all = 4800
            # elif npoints == 8192:
            #     point_all = 8192
            # else:
            #     raise NotImplementedError()
                
            # if points_raw.size(1) < point_all:
            #     point_all = points_raw.size(1)

            # fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            # local_pred = []

            # for kk in range(times):
            #     fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
            #     points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
            #                                             fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                # points = test_transforms(points)

            if base_model.__class__.__name__ == 'ModelProject':
                latent_point = base_model(points.permute(0,2,1).contiguous())
            else:
                logits, _, _ = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)

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

    latent_point_all = []
    labels_all = []

    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        
        # img = img.cuda().float()
        # import pdb; pdb.set_trace()
        
        points = data[0].to(args.local_rank)
        
        # import pdb; pd.set_trace()
        
        label = data[1]

        im = data[2].cuda()
       
        
        batch_size = points.shape[0]
                    
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # latent_img = clip_model.encode_image(img).float()
            if base_model.__class__.__name__ == 'ModelProject':
                latent_point = base_model(points.permute(0,2,1).contiguous())
            else:
                ret, latent_point, _ = base_model(points)

            

            ## GET TEXT FEATURES OF CAPTIONS
            text_features = clip_model.encode_text(text_validation)

            latent_img = clip_model.encode_image(im)

            # import pdb; pdb.set_trace()    
            # normalize features
            latent_point = (latent_point / latent_point.norm(dim=-1, keepdim=True))
            latent_img = (latent_img / latent_img.norm(dim=-1, keepdim=True))
            latent_point_all.append(latent_img)
            labels_all.append(label)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)  


            # import pdb; pdb.set_trace()



            # compute similarity
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * latent_point @ text_features.t().float()
            logits_per_text = logits_per_image.t()

            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            # import pdb; pdb.set_trace()


            for i in range(len(probs)):

                

                ind = np.argmax(probs[i])
                prediction = val_classes[ind]
                if prediction == val_classes[label[i]]:                        
                    acc_sh[label[i]] += 1
                    overall_acc_sh += 1
                overall_count_sh += 1
                acc_count_sh[label[i]] += 1 
    
    latent_point_all = torch.cat(latent_point_all,dim=0).detach().cpu().numpy()
    labels_all = torch.cat(labels_all).detach().cpu().numpy()

    np.savez_compressed('/data/dhegde1/code/CLASP_pb/Point-BERT/experiments_v3/misc/umap/clip_post', a=latent_point_all, b=labels_all)

    import pdb; pdb.set_trace()
    for i in range(len(acc_sh)): acc_sh[i] /= acc_count_sh[i]

    acc_sh = np.mean(np.array(acc_sh))
    overall_acc_sh /= overall_count_sh

    # print_log('[MN40 Validation] EPOCH: %d  acc = %.4f' % (epoch,overall_acc_sh), logger=logger)

    if args.distributed:
        torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        if len(val_classes) == 40:
            val_writer.add_scalar('Metric/ACC_MN40', overall_acc_sh, epoch)
        else:
            val_writer.add_scalar('Metric/ACC_MN10', overall_acc_sh, epoch)



    return overall_acc_sh, acc_sh, Acc_Metric(overall_acc_sh)



def validate_MN40Fuse(args,base_model,clip_model, test_dataloader,text_validation,val_classes_dict,val_writer,epoch,logger,config):
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
        
        # img = img.cuda().float()
        
        
        points = data[0].to(args.local_rank)
        points = misc.fps(points,args.npoints)
        
        label = data[1]
        
        img = data[2].cuda()
        
        batch_size = points.shape[0]
                    
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # latent_img = clip_model.encode_image(img).float()
            if base_model.__class__.__name__ == 'ModelProject':
                latent_point = base_model(points.permute(0,2,1).contiguous())
            else:
                ret, latent_point, _ = base_model(points)

            
            

            ## GET TEXT FEATURES OF CAPTIONS
            text_features = clip_model.encode_text(text_validation)

            bs = latent_point.shape[0]
            nc = text_features.shape[0]

            logits_fuse = torch.zeros((bs,nc)).cuda()

            prob_img = []

            for i in range(3):
                latent_img = clip_model.encode_image(img[:,i,:,:])
                latent_img = latent_img / latent_img.norm(dim=-1, keepdim=True)  

                
                logits_per_image =  latent_img @ text_features.t().float()

                prob_img.append(logits_per_image.softmax(dim=-1).cpu().numpy())
                logits_fuse += logits_per_image

            
            # normalize features
            latent_point = (latent_point / latent_point.norm(dim=-1, keepdim=True))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
            
           
            logit_scale = clip_model.logit_scale.exp()
            logits_per_pts = logit_scale * latent_point @ text_features.t().float()
            logits_per_text = logits_per_pts.t()

            # probs_point = logits_per_pts.softmax(dim=-1).cpu().numpy()

            # compute image/text similarity
            

            # probs_image = logits_per_image.softmax(dim=-1).cpu().numpy()
            prob_img = np.array(prob_img)
            probs_point = logits_per_pts.softmax(dim=-1)
            probs_point = probs_point.unsqueeze(dim=0).cpu().numpy()
            probs = np.concatenate((prob_img[:2],probs_point),axis=0)
            # probs = probs.max(axis=0)
            probs_args = np.argmax(probs,axis=-1)

            import pdb; pdb.set_trace()


            # probs = prob_img[2]            
            
            # logits_fuse = logit_scale * fuse_features @ text_features.t().float()

            


            # probs = logits_fuse.softmax(dim=-1).cpu().numpy()




            for i in range(len(probs)):

                
                
                ind = np.argmax(probs[i])
                prediction = val_classes[ind]
                if prediction == val_classes[label[i]]:                        
                    acc_sh[label[i]] += 1
                    overall_acc_sh += 1
                overall_count_sh += 1
                acc_count_sh[label[i]] += 1 

    for i in range(len(acc_sh)): acc_sh[i] /= acc_count_sh[i]

    acc_sh = np.mean(np.array(acc_sh))
    overall_acc_sh /= overall_count_sh

    # print_log('[MN40 Validation] EPOCH: %d  acc = %.4f' % (epoch,overall_acc_sh), logger=logger)

    if args.distributed:
        torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        if len(val_classes) == 40:
            val_writer.add_scalar('Metric/ACC_MN40', overall_acc_sh, epoch)
        else:
            val_writer.add_scalar('Metric/ACC_MN10', overall_acc_sh, epoch)



    return overall_acc_sh, acc_sh, Acc_Metric(overall_acc_sh)


def validate_MN40Image(args,base_model,clip_model, test_dataloader,text_validation,val_classes_dict,val_writer,epoch,logger,config):
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
        
        # img = img.cuda().float()
        
        
        points = data[0].to(args.local_rank)
        # points_raw = points.to(args.local_rank)
        # if npoints == 1024:
        #     point_all = 1200
        # elif npoints == 2048:
        #     point_all = 2400
        # elif npoints == 4096:
        #     point_all = 4800
        # elif npoints == 8192:
        #     point_all = 8192
        # else:
        #     raise NotImplementedError()
            
        # if points_raw.size(1) < point_all:
        #     point_all = points_raw.size(1)

        # fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)

        # fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
        # points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
        #                                         fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

        # points = test_transforms(points)
        
        label = data[1]
        
        img = data[2][:,1,:,:].cuda()
        
        batch_size = points.shape[0]
                    
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # latent_img = clip_model.encode_image(img).float()
            if base_model.__class__.__name__ == 'ModelProject':
                latent_point = base_model(points.permute(0,2,1).contiguous())
            else:
                ret, latent_point, _ = base_model(points)

            
            latent_img = clip_model.encode_image(img)

            ## GET TEXT FEATURES OF CAPTIONS
            text_features = clip_model.encode_text(text_validation)

                
            # normalize features
            latent_point = (latent_point / latent_point.norm(dim=-1, keepdim=True))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
            latent_img = latent_img / latent_img.norm(dim=-1, keepdim=True)  



            # import pdb; pdb.set_trace()



            # # compute point/text similarity
            # logit_scale = clip_model.logit_scale.exp()
            # logits_per_pts = logit_scale * latent_point @ text_features.t().float()
            # logits_per_text = logits_per_pts.t()

            # probs_point = logits_per_pts.softmax(dim=-1).cpu().numpy()

            # compute image/text similarity
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * latent_img @ text_features.t().float()
            logits_per_text = logits_per_image.t()

            probs_image = logits_per_image.softmax(dim=-1).cpu().numpy()

            # import pdb; pdb.set_trace()


            probs = probs_image

            for i in range(len(probs)):

                

                ind = np.argmax(probs[i])
                prediction = val_classes[ind]
                if prediction == val_classes[label[i]]:                        
                    acc_sh[label[i]] += 1
                    overall_acc_sh += 1
                overall_count_sh += 1
                acc_count_sh[label[i]] += 1 

    for i in range(len(acc_sh)): acc_sh[i] /= acc_count_sh[i]

    acc_sh = np.mean(np.array(acc_sh))
    overall_acc_sh /= overall_count_sh

    # print_log('[MN40 Validation] EPOCH: %d  acc = %.4f' % (epoch,overall_acc_sh), logger=logger)

    if args.distributed:
        torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        if len(val_classes) == 40:
            val_writer.add_scalar('Metric/ACC_MN40', overall_acc_sh, epoch)
        else:
            val_writer.add_scalar('Metric/ACC_MN10', overall_acc_sh, epoch)



    return overall_acc_sh, acc_sh, Acc_Metric(overall_acc_sh)

    

if __name__ == '__main__':
    main()