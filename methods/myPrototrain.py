import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from methods.mymodeltrain import MyModel

import utils

torch.autograd.set_detect_anomaly(True)

class MyProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, embed_dim=128, margin=0.5):
        super(MyProtoNet, self).__init__(model_func,  n_way, n_support)
        self.feat_dim = embed_dim
        self.feature = MyModel(model_func, self.feat_dim)
        self.cos_min = -1.+1e-7
        self.cos_max = 1.-1e-7
        self.scale = 64

        self.margin = margin
        self.loss_fn  = nn.CrossEntropyLoss()
    
    def cos_dist_func(self, z_support, z_query):
        z_support   = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query     = z_query.contiguous().view(self.n_way * self.n_query, -1 )
        
        dist_mat = torch.mm(z_query, z_support.t()).view(self.n_way, self.n_query, self.n_way, self.n_support)
        medval, medind = torch.median(dist_mat, dim=3)
        return medval.view(-1, self.n_way)
    
    def parse_feature(self,x,is_feature):
        x    = Variable(x.to(self.device))
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            z_all       = self.feature.forward(x)
            #print(z_all)
            z_all = nn.functional.normalize(z_all, p=2, dim=1)
            #print(z_all)
            #print(z_all.shape)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query
    """
    def forward(self,x):
        out  = self.feature.forward(x)
        print('out', out)
        normalized_out = nn.functional.normalized(out, p=2, dim=1)
        print('normalize', normalized_out)
        return normalized_out
    """
    
    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)
        cos_dist = self.cos_dist_func(z_support, z_query)
        #print('[set_forward]')
        #print('cos_dist', cos_dist)
        #print('scaled cos_dist', -self.scale*cos_dist)
        return self.scale*cos_dist
    
    def add_margin(self, cos_dist):
        # add self.margin to the select column
        """
        with torch.no_grad():
            indx_0 = torch.arange(cos_dist.size(0)).cuda()
            indx_1 = torch.arange(self.n_way).repeat_interleave(self.n_query).cuda()
            picked = cos_dist.gather(1, indx_1.view(-1,1))
        """
        indx_1 = torch.arange(self.n_way).repeat_interleave(self.n_query).cuda()
        
        #print(cos_dist)
        to_add_margin = cos_dist[np.arange(0,cos_dist.size(0)),indx_1]
        angles = torch.acos(torch.clamp(to_add_margin, self.cos_min, self.cos_max))
        new_ang = angles + self.margin
        cos_dist[np.arange(0,cos_dist.size(0)),indx_1] = torch.cos(new_ang)
        #print(cos_dist)

        #cos_dist[indx_0,indx_1.view(-1,1)] = torch.cos(new_ang)
        return cos_dist
    
    def compact_loss(self, z_support):
        z_sup_mean = z_support.mean(1)
        dist_mat = torch.mm(z_sup_mean, z_sup_mean.t())
        diag = torch.diag(dist_mat)
        return diag.mean()
    
    def set_forward_loss(self, x):
        z_support, z_query  = self.parse_feature(x, False)
        cos_dist = self.cos_dist_func(z_support, z_query)
        cos_dist_with_margin = self.add_margin(cos_dist)
        
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())
        #print('[set_forward_loss]')
        #print('cos_dist_with_margin', cos_dist_with_margin)
        #print('-self.scale*cos_dist_with_margin', -self.scale*cos_dist_with_margin)
        class_loss = self.loss_fn(self.scale*cos_dist_with_margin, y_query)
        
        compact_loss = self.compact_loss(z_support)
        #return torch.add(class_loss, compact_loss)
        return class_loss + compact_loss
"""
def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
"""
