import numpy as np
import os
import glob
import argparse
import backbone

model_dict = dict(
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34)

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='miniImagenet',        help='training base model')
    parser.add_argument('--model'       , default='ResNet10',      help='backbone architecture') 
    parser.add_argument('--method'      , default='baseline',   help='baseline/protonet') 
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training')
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') 
    parser.add_argument('--freeze_backbone'   , action='store_true', help='Freeze the backbone network for finetuning') 
    parser.add_argument('--task'   , default='fsl', help='task for few-shot learninig or cross-domain few-shot learning') 
    parser.add_argument('--models_to_use', '--names-list', nargs='+', default=['miniImageNet'], help='pretained model to use')
    parser.add_argument('--fine_tune_all_models'   , action='store_true',  help='fine-tune each model before selection') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--margin'      , default=0.5, help='angle margin in myModel')
    parser.add_argument('--embed_dim'   , default=512, help='embed_dim of myModel')
    parser.add_argument('--logit_scale' , default=64 , help='logit_scale of myModel')

    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch') # for meta-learning methods, each epoch contains 100 episodes
    else:
       raise ValueError('Unknown script')
        
    return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    #print('best_file: ', best_file)
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
