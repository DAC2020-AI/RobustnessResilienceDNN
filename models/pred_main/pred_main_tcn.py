import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import operator as op

from dl_models.models.pred_main.pred_main_utils import *
from dl_models.models.pred_main.tcn import *
from dl_models.models.base import *
import h5py
import sys
import numpy as np
import pickle

class predMainTCNPT(nn.Module):
    def __init__(self,mmc=False,D=10):
        super(predMainTCNPT, self).__init__()
        
        input_size = 50
        num_channels = [25] * 2
        kernel_size = 7
        dropout = 0.05
        output_size = 1
        self.mmc = mmc
        self.class_number = 2
        if mmc:
          output_size = D
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y1 = self.tcn(x)
        o = self.linear(y1[:, :, -1])
        if self.mmc:
          # print(o.size())
          # o = torch.tensor([[1,2]*128])
          # print(o.size())
          o_expand = torch.unsqueeze(o, 1).repeat(1,self.class_number,1)
          logits = -1*torch.sum((o_expand-self.mean_logits)**2,-1)
          # print(logits)
          # exit()
          return logits
        return torch.sigmoid(o)

    def set_mean_logits(self, mean_logits):
      self.mean_logits = torch.unsqueeze(torch.from_numpy(mean_logits), 0)

class predMainTCN(ModelBase):
  def __init__(self):
    super(predMainTCN,self).__init__('predMain','tcn')

    self.param_layer_ids       = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001', '0.001','0.001', '0.001','0.001', '0.001']

    self.lr = 0.001
    self.mmc = False
    self.D = 10

  def init_mmc(self, mmc):
    self.mmc = mmc
    self.D = 10
    self.class_number = 2
    
  def build_model(self):
    model = predMainTCNPT(self.mmc,self.D)
    if self.mmc:
      # 10 is variance
      mean_logits = 10*self.calculate_mean_logits(self.class_number,self.D)
      print("mean_logits",mean_logits)
      model.set_mean_logits(mean_logits)

    self.set_model( model, self.param_layer_ids, self.default_prune_factors )

  def load_dataset(self, ):
    train_set, test_data= load_dataset()
    
    self.set_data(train_set, test_data, test_data)

  def compile_model(self, loss='binary_crossentropy', optimizer='adam', metrics=None):
    if self.mmc:
      super().compile_model(loss="mmc_loss", metrics=metrics, optimizer=optimizer)
      # super().compile_model(loss="categorical_crossentropy", metrics=metrics, optimizer=optimizer)
    else:
      super().compile_model(loss=loss, metrics=metrics, optimizer=optimizer)

  def robust_error(self):
    return self.check_model(self.adversarial_test)
  
  def test_error(self):
    return self.check_model(self.original_test)

  def robust_loss(self):
    return self.check_model_loss(self.adversarial_test)
  
  def test_loss(self):
    return self.check_model_loss(self.original_test)