import os 
import pandas as pd 

from glob import glob

def img_to_csv(path):

    # Create an empty dataset
    data = {'filename': [], 'class': [], 'class_num': []}

    # iterate through the direcory and append to data
    for file in os.listdir(path):
        name, ext = os.path.splitext(file)
        item, num = name.split('_')
        data['filename'].append(file)
        data['class'].append(item)
        if item == 'bolt':
            data['class_num'].append(0)
        elif item == 'flange':
            data['class_num'].append(1)
        elif item == 'lead-block':
            data['class_num'].append(2)
        elif item == 'nut':
            data['class_num'].append(3)
        elif item == 'pipe':
            data['class_num'].append(4)
    
    # Create a DataFrame
    csv_df = pd.DataFrame(data)

    return csv_df


def main():
    data_path = 'dataset/set_dataset'
    #for folder in ['train', 'val','test']:
    for folder in ['train', 'val', 'test']:
        image_path = os.path.join(data_path, folder)
        data_frame = img_to_csv(image_path)
        save_to = os.path.join(data_path)
        data_frame.to_csv((save_to + f'{folder}_labes.csv'), index=None)
        print(f"Successfully save image data to {save_to}")

if __name__ == "__main__":
    main()