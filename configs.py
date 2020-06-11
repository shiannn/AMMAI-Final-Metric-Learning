import os
input_dir = '/tmp2/b05902020'

save_dir = os.path.join(input_dir, 'logs')
model_path = None

miniImageNet_dir = os.path.join(input_dir, 'miniImagenet')
miniImageNet_path = os.path.join(miniImageNet_dir, 'data')
miniImageNet_val_path = os.path.join(miniImageNet_dir, 'val')
miniImageNet_test_path = os.path.join(miniImageNet_dir, 'test')

ISIC_path = "./filelists/ISIC"
EuroSAT_path = "./filelists/EuroSAT/2750"
