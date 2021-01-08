import numpy as np
import pandas as pd
from glob import glob
import shutil, os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import argparse
import yaml

# Argument parser
parser = argparse.ArgumentParser(description='VinAI Big Data Chest X-ray Anormalitles Detection')
parser.add_argument('--input', type=str, help='Input of preprocessed image data folder')
parser.add_argument('--ratio', type=float, default=0.8, help='Train/val split ratio.')
args = parser.parse_args()

if __name__ == '__main__':
    # Read training CSV file
    train_df = pd.read_csv(f'{args.input}/train.csv')
    train_df['image_path'] = f'{args.input}/train/' + train_df.image_id + '.png'
    train_df = train_df[train_df.class_id!=14].reset_index(drop = True)

    # Group to fold based on ratio
    # e.g 80% training and 20% testing, ratio = 0.8
    # n_splits will equals to 1/(1-ratio) = 5
    # because we take 1 fold for validating
    # and 4 folds for training
    gkf  = GroupKFold(n_splits = int(1/(1-args.ratio)))
    train_df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups = train_df.image_id.tolist())):
        train_df.loc[val_idx, 'fold'] = fold

    train_files = []
    val_files   = []
    val_files += list(train_df[train_df.fold==0].image_path.unique())
    train_files += list(train_df[train_df.fold!=0].image_path.unique())

    # Create train/val folder if not exist
    os.makedirs('./vinbigdata', exist_ok=True)
    os.makedirs('./vinbigdata/labels/train', exist_ok=True)
    os.makedirs('./vinbigdata/labels/val', exist_ok=True)
    os.makedirs('./vinbigdata/images/train', exist_ok=True)
    os.makedirs('./vinbigdata/images/val', exist_ok=True)

    # Copy images and labels file for train/val sets
    label_dir = './labels'
    for file in tqdm(train_files):
        shutil.copy(file, './vinbigdata/images/train')
        filename = file.split('/')[-1].split('.')[0]
        shutil.copy(os.path.join(label_dir, filename+'.txt'), './vinbigdata/labels/train')
        
    for file in tqdm(val_files):
        shutil.copy(file, './vinbigdata/images/val')
        filename = file.split('/')[-1].split('.')[0]
        shutil.copy(os.path.join(label_dir, filename+'.txt'), './vinbigdata/labels/val')
    
    # Create necessary files for training yolo
    class_ids, class_names = list(zip(*set(zip(train_df.class_id, train_df.class_name))))
    classes = list(np.array(class_names)[np.argsort(class_ids)])
    classes = list(map(lambda x: str(x), classes))

    cwd = './'

    with open(join( cwd , 'train.txt'), 'w') as f:
        for path in glob('./vinbigdata/images/train/*'):
            f.write(path+'\n')
                
    with open(join( cwd , 'val.txt'), 'w') as f:
        for path in glob('./vinbigdata/images/val/*'):
            f.write(path+'\n')

    data = dict(
        train =  join( cwd , 'train.txt') ,
        val   =  join( cwd , 'val.txt' ),
        nc    = 14,
        names = classes
    )

    with open(join( cwd , 'vinbigdata.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)