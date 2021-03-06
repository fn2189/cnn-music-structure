import os
import torch.utils.data
import torchvision

import numpy as np

class DictDataset(torch.utils.data.Dataset):
    '''
    Loads dataset from a dictionary of the format: 
    
    dict = { 'train' : [X_path, X_shape, y_path, y_shape], 'val': ..., 'train': ....,}
    
    where fname is a filename of an image and label is a number or
    array that annotates the image
    '''

    def __init__(self, data, split):
        """
        data: a dictionnary formatted like above
        split: the split to consider
        """

        self.data = data
        self.split = split

    def __len__(self):
        return len(self.data[self.split])

    def __getitem__(self, idx):
        X_path = self.data[self.split][idx][0]
        X_shape = self.data[self.split][idx][1]
        y_path = self.data[self.split][idx][2]
        y_shape = self.data[self.split][idx][3]
        frame_idx = self.data[self.split][idx][4]

        #print('X shape: ', X_shape, ', frame_idx: ', frame_idx)
        
        X_train = np.memmap(
            X_path,
            dtype='float32',
            mode='w+',
            shape=tuple(X_shape)
            )[frame_idx, :, :, :]
        X = torch.from_numpy(X_train)
        del X_train
        
        y_train = np.memmap(
            y_path,
            dtype='float32',
            mode='w+',
            shape=tuple(y_shape)
            )[frame_idx, :]
        y = torch.from_numpy(y_train)
        del y_train
        
        return X, y
