import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import operator as op

from dl_models.models.us_highway.us_highway_utils import *
from dl_models.models.base import *
import h5py
import sys
import numpy as np
import pickle

class usHighWayLSTMPT(nn.Module):
    def __init__(self,predict_seconds):
        super(usHighWayLSTMPT, self).__init__()
        self.LSTM_feat = 256
        self.seq_size = 100
        self.features  = 59
        self.nb_classes = 2*predict_seconds
    
        self.LSTM = nn.LSTM(self.features, self.LSTM_feat)
        self.fc1 = nn.Linear(self.seq_size * self.LSTM_feat, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(132, self.nb_classes)

    def forward(self, x):
        batch_size = x.size(0)   

        hidden = [Variable(torch.rand(1, batch_size, self.LSTM_feat, device = x.device)),
                    Variable(torch.rand(1, batch_size, self.LSTM_feat, device = x.device))]

        self.LSTM.flatten_parameters()
        x = x.view(-1, self.seq_size,self.features)
        featrues = x[:,0,:].narrow(1,0,4)
        x = x.permute([1,0,2])      #Move to seq-batch-samp
        output = []
        for i in range(self.seq_size):   #Loop over TIME dimension to allow hidden layer injects
            out, hidden = self.LSTM(x[i].unsqueeze(0), hidden)
            output.append(out.squeeze(0))
        out = torch.stack(output)
        out = out.permute([1,0,2])  #Return to batch-seq-samp
        out = out.contiguous().view(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # append first four features
        out = torch.cat((out, featrues), 1)
        out = self.fc3(out)
        return out

class usHighWayLSTM(ModelBase):
  def __init__(self):
    super(usHighWayLSTM,self).__init__('usHighWay','lstm')

    self.param_layer_ids       = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001', '0.001','0.001', '0.001','0.001', '0.001']

    self.lr = .000001

  def build_model(self,predict_seconds=10):
    model = usHighWayLSTMPT(predict_seconds)
    
    self.set_model( model, self.param_layer_ids, self.default_prune_factors )

  def load_dataset(self, ):
    train_set, test_data= load_dataset()
    
    self.set_data(train_set, test_data, test_data)

  def compile_model(self, loss='mse', optimizer='sgd', metrics=None):
    metrics = ["mse"]
    super().compile_model(loss=loss, metrics=metrics)
    self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9, nesterov=False)

  def robust_error(self):
    return self.check_model(self.adversarial_test)
  
  def test_error(self):
    return self.check_model(self.original_test)

  def robust_loss(self):
    return self.check_model_loss(self.adversarial_test)
  
  def test_loss(self):
    return self.check_model_loss(self.original_test)