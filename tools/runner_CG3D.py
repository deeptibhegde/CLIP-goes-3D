import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter


import random

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms


from models.PointMLP import pointMLPProject as pointMLP
from models.PointConv import PointConvDensityClsSsgProject as PointConv
from utils.contrastive_loss import ContrastiveLoss, BarlowTwins

import math 
import os
# from ..SLIP.models import CLIP as SLIP
# import ..SLIP.losses 

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

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),(_, MN40_dataloader,mn40_classes),(_, MN10_dataloader,mn10_classes), \
    (_, scan_dataloader,scan_classes),(s3_sampler,s3_dataloader,s3_classes) = \
                                                            builder.dataset_builder_clasp(args, config.dataset.train,real=args.real), \
                                                            builder.dataset_builder_clasp(args, config.dataset.train,train=False,real=args.real), \
                                                            builder.dataset_builder(args, config.dataset.mn40), \
                                                            builder.dataset_builder(args, config.dataset.mn10), \
                                                            builder.dataset_builder(args, config.dataset.scan),\
                                                            builder.dataset_builder_S3DIS(args, config.dataset.s3dis)

    if args.scene:
        train_sampler,train_dataloader = s3_sampler,s3_dataloader

    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    
    # build model
    if config.model.NAME == 'PointMLP':
        base_model = pointMLP()
    elif config.model.NAME == 'PointConv':
        base_model = PointConv()
    else:
        base_model = builder.model_builder(config.model)

    contrastive_loss = ContrastiveLoss(config.total_bs,temperature=0.5,c=0.05,r=1.5)
    barlow_twins = BarlowTwins(config)
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    for p in base_model.parameters():
        p.requires_grad = True

    if args.use_gpu:    
        base_model.to(args.local_rank)

    # LOAD VISION LANGUAGE MODEL
    if args.clip or args.text or args.image:
        if args.VL == 'CLIP':
            clip_model, preprocess = clip.load("RN50x16",device=args.local_rank,jit=False)

        elif args.VL == 'SLIP':
            # import pdb; pdb.set_trace()
            clip_model = slip_models.SLIP_VITB16(visual_prompting=args.visual_prompting,ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim).to(args.local_rank)
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
            if args.resume and args.start_ckpts is None:
                clip_dict = torch.load(args.experiment_path + '/ckpt-last.pth')['visual_clip_model']
                clip_model.visual.load_state_dict(clip_dict)

            if args.start_ckpts is not None:
                clip_dict = torch.load(args.start_ckpts)['visual_clip_model']
                clip_model.visual.load_state_dict(clip_dict,strict=False)



       
    base_model.cuda()
    for p in base_model.parameters():
        p.requires_grad = True
    if args.clip or args.text or args.image:
        clip_model.to(args.local_rank)


    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    # params = []
    
    # for key, value in clip_model.named_parameters():        
    #     if value.requires_grad: 
    #         params.append((key, value))

    # optimizer_clip = builder.build_VPT_optimizer(params, config)
    # scheduler_clip = builder.build_VPT_scheduler(config,optimizer_clip)
    # optimizer_clip, scheduler_clip = builder.build_opti_sche(clip_model, config)

    # optimizer = builder.build_VPT_optimizer(params, config)
    # scheduler = builder.build_VPT_scheduler(config,optimizer)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)


    # PREP FOR VALIDATION ################################################################
    if args.clip or args.text or args.image:
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


    if (args.clip or args.text or args.image) and not args.scene:
        overall_acc_sh, acc_sh, metric = validate(args,base_model,clip_model,test_dataloader,text_validation,val_classes,val_writer,start_epoch,logger,config)
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

        # overall_acc_MN40_fuse, class_wise_acc_MN40_fuse, metrics_MN40_fuse = validate_MN40Fuse(args,base_model,clip_model,MN40_dataloader,mn40_text_validation,mn40_classes,val_writer,start_epoch,logger,config)
        # print_log("{{MODELNET40 FUSED overall accuracy: %.3f}}"%overall_acc_MN40_fuse,logger = logger) 
        # print_log("{{MODELNET40 FUSED class-wise mean Accuracy: %.3f}}"%overall_acc_MN40_fuse,logger = logger)
        
        overall_acc_scan, class_wise_acc_scan, metrics_scan = validate_ZS(args,base_model,clip_model,scan_dataloader,scan_text_validation,scan_classes,val_writer,start_epoch,logger,config)
        print_log("{{ScanObjectNN Validation overall accuracy: %.3f}}"%overall_acc_scan,logger = logger) 
        print_log("{{ScanObjectNN Validation class-wise mean Accuracy: %.3f}}"%class_wise_acc_scan,logger = logger)

        


    ################################################################################################

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    npoints = args.npoints

    # trainval
    # training
    base_model.zero_grad()
    # 
    clip_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        base_model.train()
        # clip_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['CLSLoss','TextLoss','ImageLoss','clip_loss'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)


        for idx, (taxonomy_ids, model_ids, points, (img,depth_img), caption, class_name,label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            # optimizer_clip.zero_grad()

            # import pdb; pdb.set_trace()

            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            points = points.cuda().float()
            label = label.cuda()
            if args.image:
                img = img.cuda()
                depth_img = depth_img[:,1,:,:].cuda()
            
            batch_size = points.shape[0]
            
            captions_tok = clip.tokenize(caption).cuda()
            # class_name = class_name.to(args.local_rank)
            # taxonomy_ids = taxonomy_ids.to(args.local_rank)
            # model_ids = model_ids.to(args.local_rank)

            
            if args.random_sample:
                num_sample = random.choice([1024,2048,8192])
                points = misc.fps(points,num_sample)
            else:
                points = misc.fps(points,args.npoints)

            points = train_transforms(points)

            # initialize loss values
            _loss = torch.tensor(0.0).to(args.local_rank)
            clip_loss = torch.tensor(0.0).to(args.local_rank)
            text_loss = torch.tensor(0.0).to(args.local_rank)
            image_loss = torch.tensor(0.0).to(args.local_rank)
            dvae_loss = torch.tensor(0.0).to(args.local_rank)
            loss = torch.tensor(0.0).to(args.local_rank)


            # import pdb; pdb.set_trace()

            

            if base_model.__class__.__name__ == 'ModelProject' or base_model.__class__.__name__ == 'PointConvDensityClsSsgProject':
                latent_point = base_model(points.permute(0,2,1).contiguous())
            else:
                ret,latent_point, _ = base_model(points)

            latent_point = latent_point.squeeze(-1)
            latent_point = latent_point / latent_point.norm(dim=-1, keepdim=True)

            if torch.isnan(latent_point).sum()>0:
                import pdb; pdb.set_trace()

        

                        
            ## GET IMAGE FEATURES OF CAPTIONS
            if args.clip or args.image:
                latent_img = clip_model.encode_image(img).float()

                if torch.isnan(latent_img).sum()>0:
                    import pdb; pdb.set_trace()
                latent_img = (latent_img / latent_img.norm(dim=-1, keepdim=True))

                if args.depth:
                    latent_depth_img = clip_model.encode_image(depth_img).float()

                    
                    latent_depth_img = (latent_depth_img / latent_depth_img.norm(dim=-1, keepdim=True))

                

            ## GET TEXT FEATURES OF CAPTIONS
            if args.clip or args.text:
                text_features = clip_model.encode_text(captions_tok)
                with torch.no_grad():
                    text_features_val = clip_model.encode_text(text_validation)

                if torch.isnan(text_features).sum()>0:
                    import pdb; pdb.set_trace()
                
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
                text_features_val = text_features_val / text_features_val.norm(dim=-1, keepdim=True)  
            

            
            # import pdb; pdb.set_trace()

            #reconstruct input
            # if args.dvae:
            #     # dvae_loss, _ = chamfer_distance(points,dvae_loss.permute(0,2,1).contiguous())

            #     _loss +=  dvae_loss   
            # compute similarity
            if args.clip:
                logit_scale = clip_model.logit_scale.exp()
                logits_per_pts = logit_scale * latent_img @ text_features.t().float()
                logits_per_text = logits_per_pts.t()

                ground_truth = torch.arange(batch_size,dtype=torch.long).to(args.local_rank)

                clip_loss = (loss_img(logits_per_pts,ground_truth)+ loss_txt(logits_per_text,ground_truth))/2

                if torch.isnan(clip_loss).sum()>0:
                    import pdb; pdb.set_trace()

                # _loss += clip_loss

            # import pdb; pdb.set_trace()

            # loss, acc = base_model.get_loss_acc(ret, label, smoothing=args.label_smoothing)
        
            # if args.cls:
            #     _loss += loss

            # TEXT / POINT 

            if args.text:
                if args.barlow:
                    text_loss = 0.01*barlow_twins(latent_point,text_features)
                else:

                    logit_scale = clip_model.logit_scale.exp()
                    logits_per_pts = logit_scale * latent_point @ text_features.t().float()
                    # logits_per_pts =  latent_point @ text_features.t().float()
                    logits_per_text = logits_per_pts.t()

                    
                    ground_truth = torch.arange(batch_size,dtype=torch.long).to(args.local_rank)

                    text_loss = (loss_img(logits_per_pts,ground_truth)+ loss_txt(logits_per_text,ground_truth))/2

                # _loss += text_loss
                if torch.isnan(text_loss).sum()>0:
                    import pdb; pdb.set_trace()

            # IMAGE / POINT
            if args.image:
                if args.barlow:
                    image_loss = 0.01*barlow_twins(latent_point,latent_img)
                else:
                    logit_scale = clip_model.logit_scale.exp()


                    logits_per_pts = logit_scale * latent_point @ latent_img.t().float()
                    logits_per_img = logits_per_pts.t()

                    

                    ground_truth = torch.arange(batch_size,dtype=torch.long).to(args.local_rank)

                    image_loss = (loss_img(logits_per_pts,ground_truth) + loss_txt(logits_per_img,ground_truth))/2
                    # image_loss = barlow_twins(latent_point,latent_img)
                # _loss += image_loss

                if torch.isnan(image_loss).sum()>0:
                    import pdb; pdb.set_trace()

            if args.depth:
                logit_scale = clip_model.logit_scale.exp()


                logits_per_pts_depth = logit_scale * latent_point @ latent_depth_img.t().float()
                logits_per_img_depth = logits_per_pts.t()

                

                ground_truth = torch.arange(batch_size,dtype=torch.long).to(args.local_rank)

                depth_loss = (loss_img(logits_per_pts_depth,ground_truth) + loss_txt(logits_per_img_depth,ground_truth))/2

            ##########################################################################

            
            if args.text:
                _loss += text_loss
            if args.image:
                _loss += image_loss
            if args.depth:
                _loss += depth_loss

            _loss.backward()
            if config.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            base_model.zero_grad()
            
            

            # forward
            # if num_iter == config.step_per_update:
            #     if config.get('grad_norm_clip') is not None:
            #         torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
            #     num_iter = 0
            #     optimizer.step()
            #     base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(),text_loss.item(),image_loss.item(),clip_loss.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss_CLIP', clip_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/text_loss', text_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/image_loss', image_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/dvae_loss', dvae_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)
                # train_writer.add_scalar('Loss/Batch/TrainACC', acc.item(), n_itr)

                # train_writer.add_scalar('Loss/Batch/CLS', loss.item(), n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 10 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            
            if scheduler is not None:
                scheduler.step(epoch)
                
            
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_2', losses.avg(1), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            # metrics = validate_(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
            # overall_acc_MN40, class_wise_acc_MN40, metrics_MN40 = validate_ZS(args,base_model,clip_model,MN40_dataloader,mn40_text_validation,mn40_classes,val_writer,epoch,logger,config)
            # print_log("{{MODELNET40 Validation overall accuracy: %.3f}}"%overall_acc_MN40,logger = logger) 
            # print_log("{{MODELNET40 Validation class-wise mean Accuracy: %.3f}}"%class_wise_acc_MN40,logger = logger)
            
            if (args.clip or args.text or args.image) and not args.scene:
                overall_acc_sh, acc_sh, metrics = validate(args,base_model,clip_model,test_dataloader,text_validation,val_classes,val_writer,epoch,logger,config)
                print_log("{{SHAPENET Validation overall accuracy: %.3f}}"%overall_acc_sh,logger = logger) 
                print_log("{{SHAPENET Validation class-wise mean Accuracy: %.3f}}"%acc_sh,logger = logger)

                overall_acc_im, acc_im, metric_im = validate_image(args,base_model,clip_model,test_dataloader,text_validation,val_classes,val_writer,epoch,logger)
                print_log("{{SHAPENET IMAGE Validation overall accuracy: %.3f}}"%overall_acc_im,logger = logger) 
                print_log("{{SHAPENET IMAGE Validation class-wise mean Accuracy: %.3f}}"%acc_im,logger = logger)

                overall_acc_MN40, class_wise_acc_MN40, metrics_MN40 = validate_ZS(args,base_model,clip_model,MN40_dataloader,mn40_text_validation,mn40_classes,val_writer,epoch,logger,config)
                print_log("{{MODELNET40 Validation overall accuracy: %.3f}}"%overall_acc_MN40,logger = logger) 
                print_log("{{MODELNET40 Validation class-wise mean Accuracy: %.3f}}"%class_wise_acc_MN40,logger = logger)


                overall_acc_MN10, class_wise_acc_MN10, metrics_MN10 = validate_ZS(args,base_model,clip_model,MN10_dataloader,mn10_text_validation,mn10_classes,val_writer,epoch,logger,config)
                print_log("{{MODELNET10 Validation overall accuracy: %.3f}}"%overall_acc_MN10,logger = logger) 
                print_log("{{MODELNET10 Validation class-wise mean Accuracy: %.3f}}"%class_wise_acc_MN10,logger = logger)
            
                # overall_acc_MN40_fuse, class_wise_acc_MN40_fuse, metrics_MN40_fuse = validate_MN40Fuse(args,base_model,clip_model,MN40_dataloader,mn40_text_validation,mn40_classes,val_writer,epoch,logger,config)
                # print_log("{{MODELNET40 FUSED overall accuracy: %.3f}}"%overall_acc_MN40_fuse,logger = logger) 
                # print_log("{{MODELNET40 FUSED class-wise mean Accuracy: %.3f}}"%overall_acc_MN40_fuse,logger = logger)

                overall_acc_scan, class_wise_acc_scan, metrics_scan = validate_ZS(args,base_model,clip_model,scan_dataloader,scan_text_validation,scan_classes,val_writer,epoch,logger,config)
                print_log("{{ScanObjectNN Validation overall accuracy: %.3f}}"%overall_acc_scan,logger = logger) 
                print_log("{{ScanObjectNN Validation class-wise mean Accuracy: %.3f}}"%class_wise_acc_scan,logger = logger)
            
            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger,clip_model=clip_model)
            if metrics.acc > 91.5 or (better and metrics.acc > 90):
                metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                if metrics_vote.better_than(best_metrics_vote):
                    best_metrics_vote = metrics_vote
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger,clip_model=clip_model)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger,clip_model=clip_model)      
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger,clip_model=clip_model)     
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()



def validate(args,base_model,clip_model,test_dataloader,text_validation,val_classes,val_writer,epoch,logger,config):
    npoints = args.npoints
    overall_acc_sh = 0
    overall_count_sh = 0
    acc_sh = [0]*len(val_classes)
    acc_count_sh = [0]*len(val_classes)
    base_model.eval()
    for idx, (taxonomy_ids, model_ids, points, (img,depth_img), caption, class_name,label) in enumerate(test_dataloader):
        
        # img = img.cuda().float()
        
        # fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
        # fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        # points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

        
        points = points.to(args.local_rank)
        
        # points = misc.fps(points,args.npoints)

        points = test_transforms(points)
        points = points.cuda()

        points = misc.fps(points,args.npoints)
       
        
        batch_size = img.shape[0]
                    
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # latent_img = clip_model.encode_image(img).float()
            
            # import pdb; pdb.set_trace()
            if base_model.__class__.__name__ == 'ModelProject' or base_model.__class__.__name__ == 'PointConvDensityClsSsgProject':
                latent_point = base_model(points.permute(0,2,1).contiguous())
            else:
                ret, latent_point, _ = base_model(points)

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
    for idx, (taxonomy_ids, model_ids, points, (img,depth_img), caption, class_name,label) in enumerate(test_dataloader):
        
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
    npoints = args.npoints
    with torch.no_grad():
        
        for idx, (taxonomy_ids, model_ids, points, (img,depth_img), caption, class_name,label) in enumerate(test_dataloader):
            points = points.to(args.local_rank)
            label = label.to(args.local_rank)

            points = misc.fps(points, npoints)

            

            if base_model.__class__.__name__ == 'ModelProject' or base_model.__class__.__name__ == 'PointConvDensityClsSsgProject':
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
    npoints = args.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, points, (img,depth_img), caption, class_name,label) in enumerate(test_dataloader):
            points_raw = points[0].to(args.local_rank)
            label = label[1].to(args.local_rank)
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

            if base_model.__class__.__name__ == 'ModelProject' or base_model.__class__.__name__ == 'PointConvDensityClsSsgProject':
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
    npoints = args.npoints
    # val_classes = val_classes.keys()
    val_classes = [key for key in val_classes_dict]
    # import pdb; pdb.set_trace()
    acc_sh = [0]*len(val_classes)
    acc_count_sh = [0]*len(val_classes)
    base_model.eval()
    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        
        # img = img.cuda().float()
        # import pdb; pdb.set_trace()
        
        points = data[0].to(args.local_rank)

        points = misc.fps(points,args.npoints)
        
        label = data[1]
       
        
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

                
            # normalize features
            latent_point = (latent_point / latent_point.norm(dim=-1, keepdim=True))
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

            fuse_features = (latent_img + latent_point)/2

                
            # normalize features
            latent_point = (latent_point / latent_point.norm(dim=-1, keepdim=True))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
            latent_img = latent_img / latent_img.norm(dim=-1, keepdim=True)  
            fuse_features = fuse_features / fuse_features.norm(dim=-1, keepdim=True)  



            # import pdb; pdb.set_trace()



            # compute point/text similarity
            logit_scale = clip_model.logit_scale.exp()
            logits_per_pts = logit_scale * latent_point @ text_features.t().float()
            logits_per_text = logits_per_pts.t()

            # probs_point = logits_per_pts.softmax(dim=-1).cpu().numpy()

            # compute image/text similarity
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * latent_img @ text_features.t().float()
            logits_per_text = logits_per_image.t()



            # probs_image = logits_per_image.softmax(dim=-1).cpu().numpy()

            # import pdb; pdb.set_trace()
            logits_fuse = 0.4*logits_per_image + 0.6*logits_per_pts
            # logits_fuse = logit_scale * fuse_features @ text_features.t().float()




            probs = logits_fuse.softmax(dim=-1).cpu().numpy()

            


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

        val_writer.add_scalar('Metric/ACC_MN40_Fuse', overall_acc_sh, epoch)

           



    return overall_acc_sh, acc_sh, Acc_Metric(overall_acc_sh)



def validate_MN40(args,base_model,clip_model, test_dataloader,text_validation,val_classes_dict,val_writer,epoch,logger,config):
    overall_acc_sh = 0
    overall_count_sh = 0
    npoints = args.npoints
    # val_classes = val_classes.keys()
    val_classes = [key for key in val_classes_dict]
    # import pdb; pdb.set_trace()
    acc_sh = [0]*len(val_classes)
    acc_count_sh = [0]*len(val_classes)
    base_model.eval()
    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        
        # img = img.cuda().float()
        
        # import pdb; pdb.set_trace()
        
        points = data[0].to(args.local_rank)
        
        points = misc.fps(points,args.npoints)
        points = test_transforms(points)
        
        label = data[1]
       
        
        batch_size = points.shape[0]
                    
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # latent_img = clip_model.encode_image(img).float()
            if base_model.__class__.__name__ == 'ModelProject' or base_model.__class__.__name__ == 'PointConvDensityClsSsgProject':
                latent_point = base_model(points.permute(0,2,1).contiguous())
            else:
                ret, latent_point, _ = base_model(points)

            

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

    
