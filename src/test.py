class FlowerDataset(Dataset):
    """
    Custom PyTorch Dataset Class to facilitate loading data for the Image Classifcation Task
    """
    def __init__(self, annotations, train_test_valid_split, mapping = None, mode = 'train', transform = None):
        """ 
        Args:
            annotations: The path to the annotations CSV file. Format: file_name, class_name
            train_test_valid_split: The path to the tags CSV file for train, test, valid split. Format: file_name, tag
            mapping: a dictionary containing mapping of class name and class index. Format : {'class_name' : 'class_index'}, Default: None
            mode: Mode in which to instantiate class. Default: 'train'
            transform: The transforms to be applied to the image data

        Returns:
            image : Torch Tensor, label_tensor : Torch Tensor, file_name : str
        """

        my_data = pd.read_csv(annotations, index_col='file_name')
        my_data['tag'] = pd.read_csv(train_test_valid_split, index_col='file_name')
        my_data = my_data.reset_index()

        self.mapping = mapping
        self.transform = transform
        self.mode = mode

        my_data = my_data.loc[my_data['tag'] == mode].reset_index(drop=True)
        self.data = my_data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mapping is not None:
            labels = int(self.mapping[self.data.loc[idx, 'class_name'].lower()])
        else:
            labels = int(self.data.loc[idx, 'class_name'])

        im_path = self.data.loc[idx, 'file_name']

        label_tensor =  torch.as_tensor(labels, dtype=torch.long)
        im = Image.open(im_path)

        if self.transform:
            im = self.transform(im)

        if self.mode == 'test':
            # For saving the predictions, the file name is required
            return {'im' : im, 'labels': label_tensor, 'im_name' : self.data.loc[idx, 'file_name']}
        else:
            return {'im' : im, 'labels' : label_tensor}