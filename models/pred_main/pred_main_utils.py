import pandas as pd
import numpy as np
from sklearn import preprocessing
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

import h5py

predMain_dataset_path = './data/pred_main/'

class predMainDataset(Dataset):
  def __init__(self, X,y):
    self.X = X
    self.y = y
    
  def __len__(self):
    return len(self.y)
  
  def __getitem__(self, i):
    return self.X[i], self.y[i]
    
def get_dataset(data_list_path,fname):
  return predMainDataset(data_list_path,fname)


def gen_sequence(id_df, seq_length, seq_cols):
  data_matrix = id_df[seq_cols].values
  num_elements = data_matrix.shape[0]
  for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
    yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
  data_matrix = id_df[label].values
  num_elements = data_matrix.shape[0]
  return data_matrix[seq_length:num_elements, :]


def read_predMain(fname):
  # read data
  train_df = pd.read_csv(fname+'PM_train.txt', sep=" ", header=None)
  train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
  train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                      's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                      's15', 's16', 's17', 's18', 's19', 's20', 's21']

  train_df = train_df.sort_values(['id','cycle'])

  test_df = pd.read_csv(fname+'PM_test.txt', sep=" ", header=None)
  test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
  test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                      's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                      's15', 's16', 's17', 's18', 's19', 's20', 's21']

  truth_df = pd.read_csv(fname+'PM_truth.txt', sep=" ", header=None)
  truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

  # train preprocess
  rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
  rul.columns = ['id', 'max']
  train_df = train_df.merge(rul, on=['id'], how='left')
  train_df['RUL'] = train_df['max'] - train_df['cycle']
  train_df.drop('max', axis=1, inplace=True)

  w1 = 30
  w0 = 15
  train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
  train_df['label2'] = train_df['label1']
  train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

  train_df['cycle_norm'] = train_df['cycle']
  cols_normalize = train_df.columns.difference(['id','cycle','RUL','label1','label2'])
  min_max_scaler = preprocessing.MinMaxScaler()
  norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                              columns=cols_normalize, 
                              index=train_df.index)
  join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
  train_df = join_df.reindex(columns = train_df.columns)

  # test preprocess
  test_df['cycle_norm'] = test_df['cycle']
  norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                              columns=cols_normalize, 
                              index=test_df.index)
  test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
  test_df = test_join_df.reindex(columns = test_df.columns)
  test_df = test_df.reset_index(drop=True)

  rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
  rul.columns = ['id', 'max']
  truth_df.columns = ['more']
  truth_df['id'] = truth_df.index + 1
  truth_df['max'] = rul['max'] + truth_df['more']
  truth_df.drop('more', axis=1, inplace=True)

  test_df = test_df.merge(truth_df, on=['id'], how='left')
  test_df['RUL'] = test_df['max'] - test_df['cycle']
  test_df.drop('max', axis=1, inplace=True)

  test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
  test_df['label2'] = test_df['label1']
  test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2

  sequence_length = 50

  sensor_cols = ['s' + str(i) for i in range(1,22)]
  sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
  sequence_cols.extend(sensor_cols)

  seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols)) 
            for id in train_df['id'].unique())

  seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

  label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['label1']) 
             for id in train_df['id'].unique()]
  label_array = np.concatenate(label_gen).astype(np.float32)

  train_data = predMainDataset(seq_array,label_array)

  seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-sequence_length:] 
                       for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]

  seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

  y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]
  label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
  label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)

  # train_data = predMainDataset(seq_array,label_array)
  # test_data = predMainDataset(seq_array_test_last,label_array_test_last)

  # train size: 14000
  # test size: 1724
  # TODO random selection

  train_size = 14000
  train_input_np = seq_array[:train_size]
  test_input_np = np.concatenate((seq_array[train_size:],seq_array_test_last),axis=0)
  train_label_np = label_array[:train_size].astype(np.float32)
  test_label_np = np.concatenate((label_array[train_size:],label_array_test_last),axis=0).astype(np.float32)

  train_data = predMainDataset(train_input_np,train_label_np)
  test_data = predMainDataset(test_input_np,test_label_np)

  return train_data, test_data

def load_dataset():
  return read_predMain(predMain_dataset_path)



    
