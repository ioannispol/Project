import os
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#from skimage import io
from PIL import Image

"""Basic Dataset class which loads images from a folder

Returns:
    [type]: [description]
"""

class ImageDataset(Dataset):
    """Project Image dataset"""

    def __init__(self, csv_file, root_dir, transform):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        #image = Image.fromarray(image)
        y_label = torch.tensor(int(self.annotations.iloc[index, 2]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
