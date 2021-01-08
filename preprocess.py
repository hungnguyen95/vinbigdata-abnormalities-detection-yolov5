###############################################################################
# This preprocess require original dataset (dicom file)                       #
# If you haven't downloaded dataset, please download it                       #
# Original dataset Kaggle URL:                                                #
# https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data #
###############################################################################

import os
import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from tqdm.auto import tqdm
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='VinAI Big Data Chest X-ray Anormalitles Detection')
parser.add_argument('--input', type=str, help='Input Folder (Original dicom image file)')
parser.add_argument('--output', type=str, help='Output Folder (Convert to PNG image and resize)')
parser.add_argument('--resize', type=int, default=512, help='Size of image after convert to PNG')
args = parser.parse_args()

def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im

if __name__ == '__main__':
    train_df = pd.read_csv(f'{args.input}/train.csv')
    train_df['width'] = 0
    train_df['height'] = 0

    for split in ['train', 'test']:
        load_dir = f'{args.input}/{split}/'
        save_dir = f'{args.output}/{split}/'

        os.makedirs(save_dir, exist_ok=True)

        for file in tqdm(os.listdir(load_dir)):
            # set keep_ratio=True to have original aspect ratio
            xray = read_xray(load_dir + file)
            im = resize(xray, size=256)  
            im.save(save_dir + file.replace('dicom', 'png'))
            
            if split == 'train':
                image_id = file.split('.')[0]
                train_df.loc[train_df.image_id == image_id, 'height'] = xray.shape[0]
                train_df.loc[train_df.image_id == image_id, 'width'] = xray.shape[1]
    
    train_df.save(f'{args.output}/train.csv')