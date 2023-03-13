'''
References: SimCLR: 
            SupContrast: 
'''
from __future__ import print_function
import os
import sys
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pdb

import torch.nn.functional as F
# import geoopt



class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5,c=0.05,r=1.25):
        super().__init__()
        # batch_size *= 8
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.device = torch.cuda.set_device(0)#"cuda:0" if torch.cuda.is_available() else "cpu"
        self.temp = 1.0
        self.c = c 
        self.r = r
        # self.manifold = geoopt.PoincareBall(c=c)

        self.BCEWithLogits = torch.nn.BCEWithLogitsLoss()

    
    def forward(self, emb_i, emb_j,labels=None,hyperbolic=False):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper

        IMP: in hyperbolic case, we are assuming emb_i is the image feature vector 
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        # z_i = emb_i
        # z_j = emb_j
        

        representations = torch.cat([z_i, z_j], dim=0)

        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        self.negatives_mask = self.negatives_mask.to('cuda')
        self.temperature = self.temperature.to('cuda')

        nominator = torch.exp(positives / self.temperature)
        try:
            denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        except:
            import pdb; pdb.set_trace()
        
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        if loss.isnan().sum() > 0:
            import pdb; pdb.set_trace()
        return loss


class BarlowTwins(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.lambd = 0.0051
        self.config = config
        # self.args = args
        # self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        # self.backbone.fc = nn.Identity()

        # projector
        # sizes = [2048] + list(map(int, args.projector.split('-')))
        # layers = []
        # for i in range(len(sizes) - 2):
        #     layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        #     layers.append(nn.BatchNorm1d(sizes[i + 1]))
        #     layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        # self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        # self.bn = nn.BatchNorm1d(512, affine=False)

    def forward(self, z1, z2):
        # z1 = self.projector(self.backbone(y1))
        # z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        # c.div_(self.config.total_bs)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BCELoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5,c=0.05,r=1.25):
        super().__init__()
        # batch_size *= 8
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.device = torch.cuda.set_device(0)#"cuda:0" if torch.cuda.is_available() else "cpu"
        self.temp = 1.0
        self.c = c 
        self.r = r
        # self.manifold = geoopt.PoincareBall(c=c)

        self.BCEWithLogits = torch.nn.BCEWithLogitsLoss()

    def lorenz_factor(self,x, *, c=1.0, dim=-1, keepdim=False):
        """
        Parameters
        ----------
        x : tensor
            point on Klein disk
        c : float
            negative curvature
        dim : int
            dimension to calculate Lorenz factor
        keepdim : bool
            retain the last dim? (default: false)

        Returns
        -------
        tensor
            Lorenz factor
        """
        return 1 / torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Project a point from Klein model to Poincare model
    def k2p(self,x, c):
        denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
        return x / denom

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Project a point from Poincare model to Klein model
    def p2k(self,x, c):
        denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
        return 2 * x / denom

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def poincare_mean(self,x, dim=0, c=1.0):
        # To calculate the mean, another model of hyperbolic space named Klein model is used.
        # 1. point is projected from Poincare model to Klein model using p2k, output x is a point in Klein model
        x = self.p2k(x, c)
        # 2. mean is calculated
        lamb = self.lorenz_factor(x, c=c, keepdim=True)
        mean = torch.sum(lamb * x, dim=dim, keepdim=True) / torch.sum(lamb, dim=dim, keepdim=True)
        # 3. Mean is projected from Klein model to Poincare model
        mean = self.k2p(mean, c)
        return mean.squeeze(dim)


    def clip_value(self,input_vector, r):
        input_norm = torch.norm(input_vector, dim = -1)
        clip_value = float(r)/input_norm
        min_norm = torch.clamp(float(r)/input_norm, max = 1)
        return min_norm[:, None] * input_vector 

    def forward(self, emb_i, emb_j,labels=None,hyperbolic=False):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper

        IMP: in hyperbolic case, we are assuming emb_i is the image feature vector 
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        # z_i = emb_i
        # z_j = emb_j
        

        representations = torch.cat([z_i, z_j], dim=0)
        if hyperbolic:  
            labels_proto = torch.unique(labels)
            num_classes = len(labels_proto)
            class_prototypes = torch.zeros((num_classes,z_i.shape[-1]))
            # import pdb; pdb.set_trace()
            for i in labels:
                ind = labels_proto == i 
                ind = ind.nonzero().item()
                class_prototypes[ind] = self.poincare_mean(z_i[labels == i],dim=0, c=self.manifold.c.item())
            
            labels_proto = labels_proto.unsqueeze(0)
            labels = labels.unsqueeze(0)

            

            class_prototypes = self.manifold.expmap0(class_prototypes).cuda()
            z_j = self.manifold.expmap0(z_j).cuda()

            similarity_matrix = (-self.manifold.dist(z_j[:, None, :], class_prototypes) / self.temp)
        else:
            labels_proto = torch.unique(labels)
            num_classes = len(labels_proto)
            class_prototypes = torch.zeros((num_classes,z_i.shape[-1]))
            # import pdb; pdb.set_trace()
            for i in labels:
                ind = labels_proto == i 
                ind = ind.nonzero().item()
                class_prototypes[ind] = torch.mean(z_i[labels == i],dim=0)
            
            labels_proto = labels_proto.unsqueeze(0)
            labels = labels.unsqueeze(0)

            

            # import pdb; pdb.set_trace()

            similarity_matrix = 2 - torch.cdist(z_j.cuda(),class_prototypes.cuda(),p=2) #(((z_j[:, None, :].cuda() - class_prototypes.cuda())**2).sum(dim=-1) / self.temp)

        match_matrix_bool = labels_proto == labels.t()
        match_matrix = match_matrix_bool.float().cuda()
        loss = self.BCEWithLogits(similarity_matrix,match_matrix)

        return loss

            
            # representations_clipped = self.clip_value(representations,self.r)

            # representations_hyper = self.manifold.expmap0(representations_clipped)

            # similarity_matrix = (-self.manifold.dist(representations_hyper[:, None, :], representations_hyper) / self.temp)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        # import pdb; pdb.set_trace()
        self.negatives_mask = self.negatives_mask.to('cuda')
        self.temperature = self.temperature.to('cuda')

        nominator = torch.exp(positives / self.temperature)
        try:
            denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        except:
            import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss




class ObjectLevelContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        # batch_size *= 8
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size, batch_size, dtype=bool)).float())
        self.device = torch.cuda.set_device(0)#"cuda:0" if torch.cuda.is_available() else "cpu" 
        self.match_matrix = np.zeros((batch_size,batch_size))
    def forward(self, emb_i, emb_j,labels):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        labels = labels.unsqueeze(0)

        match_matrix_bool = labels == labels.t()
        # 
        
        
        similarity_matrix = z_i@z_j.t()
        
        # similarity_matrix += 1
        # similarity_matrix /= 2

        negatives_mask = (~match_matrix_bool).float().to('cuda')
        positives_mask = (match_matrix_bool).float().to('cuda')

        self.temperature = self.temperature.to('cuda')

        positives = similarity_matrix * positives_mask
        nominator = torch.exp(positives / self.temperature)
        # import pdb; pdb.set_trace()
        
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss_partial = loss_partial[match_matrix_bool]
        # import pdb; pdb.set_trace()
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        

        return loss






class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features,features_2=None, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        elif self.contrast_mode == 'cross-modal':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss




        # import pdb; pdb.set_trace()
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss