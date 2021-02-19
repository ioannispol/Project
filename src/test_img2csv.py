import os 
import pandas as pd 

path = '../../dataset/set_dataset/val'

def img_to_csv(path):
    data = {'filename': [], 'class': []}

    for file in os.listdir(path):
        name, ext = os.path.splitext(file)
        item, num = name.split('_')
        data['filename'].append(file)
        data['class'].append(item)

    csv_df = pd.DataFrame(data)

    return csv_df

data_path = '../../dataset/set_dataset/'
data_frame = img_to_csv(path)
save_to = os.path.join(data_path)
data_frame.to_csv((save_to + f'val_labes.csv'), index=None)