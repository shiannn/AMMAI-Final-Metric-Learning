import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time, glob
from itertools import combinations

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet

from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 
from utils import *
from datasets import miniImageNet_few_shot, EuroSAT_few_shot, ISIC_few_shot

from pseudo_query_generator import PseudoQeuryGenerator

def meta_test(novel_loader, n_query = 15, pretrained_dataset='miniImageNet', freeze_backbone=False, n_pseudo=100, n_way = 5, n_support = 5): 
    correct = 0
    count = 0

    iter_num = len(novel_loader) 

    acc_all = []
    for ti, (x, y) in enumerate(novel_loader):

        ###############################################################################################
        # load pretrained model on miniImageNet
        if params.method == 'protonet':
            pretrained_model = ProtoNet(model_dict[params.model], n_way = n_way, n_support = n_support)


        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, pretrained_dataset, params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'
        checkpoint_dir += '_5way_5shot'

        params.save_iter = -1
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        else:
            modelfile   = get_best_file(checkpoint_dir)

        tmp = torch.load(modelfile)
        state = tmp['state']
        pretrained_model.load_state_dict(state)
        pretrained_model.cuda() 
        ###############################################################################################
        # split data into support set and query set
        n_query = x.size(1) - n_support
        
        x = x.cuda()
        x_var = Variable(x)

        support_size = n_way * n_support 
       
        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda()    # (25,)

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # query set
        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:])  # support set
 
        if freeze_backbone == False:
            ###############################################################################################
            # Finetune components initialization 
            pseudo_q_genrator  = PseudoQeuryGenerator(n_way, n_support,  n_pseudo)
            delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()))

            ###############################################################################################
            # finetune process 
            finetune_epoch = 100
        
            fine_tune_n_query = n_pseudo // n_way
            pretrained_model.n_query = fine_tune_n_query
            pretrained_model.train()

            z_support = x_a_i.view(n_way, n_support, *x_a_i.size()[1:])
                
            for epoch in range(finetune_epoch):
                delta_opt.zero_grad()

                # generate pseudo query images
                psedo_query_set, _ = pseudo_q_genrator.generate(x_a_i)
                psedo_query_set = psedo_query_set.cuda().view(n_way, fine_tune_n_query,  *x_a_i.size()[1:])

                x = torch.cat((z_support, psedo_query_set), dim=1)
 
                loss = pretrained_model.set_forward_loss(x)
                loss.backward()
                delta_opt.step()

        ###############################################################################################
        # inference 
        
        pretrained_model.eval()
        
        pretrained_model.n_query = n_query
        with torch.no_grad():
            scores = pretrained_model.set_forward(x_var.cuda())
        
        y_query = np.repeat(range( n_way ), n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()

        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)

        acc_all.append((correct_this/ count_this *100))        
        print("Task %d : %4.2f%%  Now avg: %4.2f%%" %(ti, correct_this/ count_this *100, np.mean(acc_all) ))
        ###############################################################################################
    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')
    task = params.task

    ##################################################################
    image_size = 224
    iter_num = 600

    n_query = max(1, int(16* params.test_n_way/params.train_n_way))
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
   

    # number of pseudo images
    n_pseudo = 100

    ##################################################################
    # loading dataset 
    pretrained_dataset = "miniImageNet"
    dataset_names = ["EuroSAT", "ISIC"]

    novel_loaders = []
    if task == 'fsl':
        freeze_backbone = True

        dataset_names = ["miniImageNet"]
        print ("Loading mini-ImageNet")
        datamgr             =  miniImageNet_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, mode="test", **few_shot_params)
        novel_loader        = datamgr.get_data_loader(aug =False)
        novel_loaders.append(novel_loader)
    else:
        freeze_backbone = params.freeze_backbone

        dataset_names = ["EuroSAT", "ISIC"]

        print ("Loading EuroSAT")
        datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_loader        = datamgr.get_data_loader(aug =False)
        novel_loaders.append(novel_loader)

        print ("Loading ISIC")
        datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_loader        = datamgr.get_data_loader(aug =False)
        novel_loaders.append(novel_loader)

    print('fine-tune: ', not freeze_backbone)
    if not freeze_backbone:
        print("n_pseudo: ", n_pseudo)

    #########################################################################
    # meta-test loop
    for idx, novel_loader in enumerate(novel_loaders):
        print (dataset_names[idx])
        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
      
        meta_test(novel_loader, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, n_pseudo=n_pseudo, **few_shot_params)
