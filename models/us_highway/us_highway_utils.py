import numpy as np
import math 
import pickle, hickle
import sys
import glob
import json
import scipy
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import os
import sys 
import h5py


usHighway_dataset_path = './data/us_highway/'

class usHighwayDataset(Dataset):
  def __init__(self, data_list_path,fname, transform=None):
    self.X = []
    self.y = []
    self.index_list = []
    for path in data_list_path:
      current_path = fname+path
      current_file = np.load(current_path,"r")
      for i, output in enumerate(current_file["outputs"]):
        self.index_list.append((current_path,i))
    #   self.X.extend(current_file["inputs"].astype('float32'))
    #   self.y.extend(current_file["outputs"].astype('float32'))
    # self.X = np.array(self.X)# torch.from_numpy(np.array(self.X)).type(torch.float)
    # self.y = np.array(self.y) # torch.from_numpy(np.array(self.y)).type(torch.float)

  def __len__(self):
    return len(self.index_list)
  
  def __getitem__(self, i):
    current = self.index_list[i]
    current_file = np.load(current[0],"r")
    inputs = current_file["inputs"][current[1]].astype('float32')
    outputs = current_file["outputs"][current[1]].astype('float32')
    return inputs, outputs
    
def get_dataset(data_list_path,fname):
  return usHighwayDataset(data_list_path,fname)


def read_usHighway(fname):
  data_list_path = os.listdir(fname)
  test_dataset_size = math.ceil(len(data_list_path)/5)
  train_dataset_size = len(data_list_path)-test_dataset_size
  
  train_data = get_dataset(data_list_path[:train_dataset_size],fname)
  test_data = get_dataset(data_list_path[train_dataset_size:],fname)
  return train_data, test_data

def load_dataset():
  return read_usHighway(usHighway_dataset_path)



    
