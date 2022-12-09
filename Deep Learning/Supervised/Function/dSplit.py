# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 21:17:55 2022

@author: Chakron.D
"""
#%% Importing the libraries

# PyTroch
from torch.utils.data import DataLoader, TensorDataset

# sckitlearn
from sklearn.model_selection import train_test_split

#%% Data split & data loader function
def dSplit(data_ts,
           y_ts,
           model_test_size=0.2,
           p_batch_size=32,
           ):
    
    # split to find array length
    data_train, data_test, y_train, y_test = train_test_split(
        data_ts, y_ts, test_size=model_test_size)

    # ---------------------------------------
    # Create Datasets, Dataloader, Batch
    # ---------------------------------------
    # convert to PyTorch Datasets
    Dataset_train = TensorDataset(data_train, y_train)
    Dataset_test = TensorDataset(data_test, y_test)
    size = [Dataset_train.tensors[0].shape, Dataset_test.tensors[0].shape]

    # finally, translate into dataloader objects
    # drop_last=True, drop the last batch if it's not full batch
    DataLoader_train = DataLoader(Dataset_train, shuffle=True, batch_size=p_batch_size, drop_last=True)
    DataLoader_test = DataLoader(Dataset_test, batch_size=Dataset_test.tensors[0].shape[0])
    
    return DataLoader_train, DataLoader_test, size