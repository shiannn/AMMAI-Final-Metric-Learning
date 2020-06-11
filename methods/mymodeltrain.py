import sys
import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, backbone_func, embed_dim=512):
        super(MyModel, self).__init__()
        
        self.final_feat_dim = embed_dim
        self.backbone = backbone_func()
        self.embedder = nn.Linear(self.backbone.final_feat_dim, self.final_feat_dim)
    
    def forward(self, x):
        out = self.backbone(x)
        return self.embedder(out)

class MyModelTrain(nn.Module):    
    def __init__(self, model_func, num_class, margin=0.5, embed_dim=512, logit_scale=64):
        super(MyModelTrain, self).__init__()
        
        self.margin = margin
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.logit_scale = logit_scale

        self.epsilon = 1e-6
        self.lower_theta = -1 + self.epsilon
        self.upper_theta = 1 - self.epsilon
        
        self.feature = MyModel(model_func, self.embed_dim)
        self.classifier = nn.Linear(self.feature.final_feat_dim, self.num_class, bias=False)
        with torch.no_grad():
            self.classifier.weight.fill_(1.)
            self.classifier.weight.copy_(nn.functional.normalize(self.classifier.weight, p=2, dim=0))

        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()
    
    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        out = nn.functional.normalize(out, p=2, dim=1) *self.logit_scale
        scores  = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        y = Variable(y.cuda())

        scores = self.forward(x)
        with torch.no_grad():
            pick_idx = torch.reshape(y, (-1,)).data
            picked = scores.data[:,pick_idx]
        
        picked /= self.logit_scale
        theta = torch.acos(torch.clamp(picked, self.lower_theta, self.upper_theta))
        theta_add = theta + self.margin
        scores.data[:,pick_idx] = torch.cos(theta_add) * self.logit_scale
        
        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))
        
        return self.loss_fn(scores, y)
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.classifier.weight.copy_(nn.functional.normalize(self.classifier.weight, p=2, dim=0))

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))
                     
    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration

