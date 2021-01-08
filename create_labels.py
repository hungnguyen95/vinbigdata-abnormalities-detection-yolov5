import pandas as pd
import os
from tqdm import tqdm
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='VinAI Big Data Chest X-ray Anormalitles Detection')
parser.add_argument('--input_csv', type=str, help='Input of preprocessed training csv file')
args = parser.parse_args()

if __name__ == '__main__':
    # Create saving labels folder if not exist
    os.makedirs('./labels', exist_ok=True)

    # Read training CSV file
    train_df = pd.read_csv(args.input_csv)

    # Normalize (x,y) coordinate to range (0,1)
    train_df['x_min'] = train_df.apply(lambda row: (row.x_min)/row.width, axis=1)
    train_df['x_max'] = train_df.apply(lambda row: (row.x_max)/row.width, axis=1)
    train_df['y_min'] = train_df.apply(lambda row: (row.y_min)/row.height, axis=1)
    train_df['y_max'] = train_df.apply(lambda row: (row.y_max)/row.height, axis=1)

    # Create center of bounding box
    train_df['x_mid'] = train_df.apply(lambda row: (row.x_max+row.x_min)/2, axis =1)
    train_df['y_mid'] = train_df.apply(lambda row: (row.y_max+row.y_min)/2, axis =1)

    # Normalize image width and height to range (0,1)
    train_df['w'] = train_df.apply(lambda row: (row.x_max-row.x_min), axis =1)
    train_df['h'] = train_df.apply(lambda row: (row.y_max-row.y_min), axis =1)

    # Because YOLO labels need class ID, center and size of bounding box
    # So we need to create features array of its
    features = ['class_id' ,'x_mid', 'y_mid', 'w', 'h']

    # Write labels
    for index, row in train_df.iterrows():
        image_id = row['image_id']
        class_id = row['class_id']
        with open(f'./labels/{image_id}.txt', 'a') as file:
            label_content = ' '.join(str(x) for x in row[features])
            file.write(label_content + '\n')