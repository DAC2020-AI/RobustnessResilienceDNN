import pandas as pd
import numpy as np
import pickle

import dl_models.models.base
from dl_models.models.base import AugmentedDataset

sequence_folder_path = "data/seq_mnist/sequences/"
output_folder_path = "data/seq_mnist/"

def data_preparation(folder_path=sequence_folder_path, output_folder_path=output_folder_path):
  # data = pd.read_csv(folder_path+"testimg-0-inputdata.txt", delim_whitespace=True, header=None)
  # data = pd.read_csv(folder_path+"testimg-9999-inputdata.txt", delim_whitespace=True, header=None)
  # data = pd.read_csv(folder_path+"trainimg-59999-inputdata.txt", delim_whitespace=True, header=None)
  # data = pd.read_csv(folder_path+"trainlabels.txt", delim_whitespace=True, header=None)
  # data = pd.read_csv(folder_path+"testlabels.txt", delim_whitespace=True, header=None)
  max_length = 120
  feature_size = 4
  options = [{
    "name": "trainimg",
    "size": 60000,
    "save_name": "training_dataset",
    "label": "trainlabels.txt"
  },
  {
    "name": "testimg",
    "size": 10000,
    "save_name": "test_dataset",
    "label": "testlabels.txt"
  }
  ]
  
  for option in options:
    dataset_list = []
    labels = pd.read_csv(folder_path+option["label"], delim_whitespace=True, header=None)
    for i in range(option["size"]):
      file_path = folder_path+option["name"]+"-"+str(i)+"-inputdata.txt"
      data = np.loadtxt(file_path,delimiter=" ")
      label = int(labels.loc[i])
      result = np.zeros((max_length,feature_size))
      result[:data.shape[0],:data.shape[1]] = data
      dataset_list.append([result,label])
    dataset = AugmentedDataset(dataset_list)
    outfile = open(output_folder_path+option["save_name"],'wb')
    pickle.dump(dataset,outfile)
    outfile.close()

data_preparation()