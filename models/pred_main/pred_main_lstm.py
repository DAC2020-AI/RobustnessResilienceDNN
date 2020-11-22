import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import operator as op

from dl_models.models.pred_main.pred_main_utils import *
from dl_models.models.base import *
import h5py
import sys
import numpy as np
import pickle

class predMainLSTMPT(nn.Module):
    def __init__(self):
        super(predMainLSTMPT, self).__init__()
        # TODO update * factor
        self.LSTM_feat = 100 * 2
        self.LSTM2_feat = 50 * 2
        self.seq_size = 50
        self.features  = 25
        self.nb_classes = 1
    
        self.LSTM1 = nn.LSTM(self.features, self.LSTM_feat)
        self.dropout1 = nn.Dropout(p=0.2)
        self.LSTM2 = nn.LSTM(self.LSTM_feat, self.LSTM2_feat)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(self.LSTM2_feat, self.nb_classes)

    def forward(self, x):
        batch_size = x.size(0)   

        hidden = [Variable(torch.rand(1, batch_size, self.LSTM_feat, device = x.device)),
                    Variable(torch.rand(1, batch_size, self.LSTM_feat, device = x.device))]
        hidden2 = [Variable(torch.rand(1, batch_size, self.LSTM2_feat, device = x.device)),
                    Variable(torch.rand(1, batch_size, self.LSTM2_feat, device = x.device))]

        self.LSTM1.flatten_parameters()
        x = x.view(-1, self.seq_size,self.features)
        x = x.permute([1,0,2])      #Move to seq-batch-samp
        output = []
        for i in range(self.seq_size):   #Loop over TIME dimension to allow hidden layer injects
            out, hidden = self.LSTM1(x[i].unsqueeze(0), hidden)
            output.append(out.squeeze(0))
        out = torch.stack(output)
        out = out.permute([1,0,2])  #Return to batch-seq-samp
        out = self.dropout1(out)
        # LSTM2
        out = out.permute([1,0,2])
        output = []
        self.LSTM2.flatten_parameters()
        for i in range(self.seq_size):   #Loop over TIME dimension to allow hidden layer injects
            out2, hidden = self.LSTM2(out[i].unsqueeze(0), hidden2)
            output.append(out2.squeeze(0))
        out = output[-1]
        out = self.dropout2(out)
        out = out.contiguous().view(batch_size, -1)

        out = F.sigmoid(self.fc1(out))
        return out

class predMainLSTM(ModelBase):
  def __init__(self):
    super(predMainLSTM,self).__init__('predMain','lstm')

    self.param_layer_ids       = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001', '0.001','0.001', '0.001','0.001', '0.001']

    self.lr = 0.001

  def build_model(self):
    model = predMainLSTMPT()
    
    self.set_model( model, self.param_layer_ids, self.default_prune_factors )

  def load_dataset(self, ):
    train_set, test_data= load_dataset()
    
    self.set_data(train_set, test_data, test_data)

  def compile_model(self, loss='binary_crossentropy', optimizer='adam', metrics=None):
    super().compile_model(loss=loss, metrics=metrics, optimizer=optimizer)

  def robust_error(self):
    return self.check_model(self.adversarial_test)
  
  def test_error(self):
    return self.check_model(self.original_test)

  def robust_loss(self):
    return self.check_model_loss(self.adversarial_test)
  
  def test_loss(self):
    return self.check_model_loss(self.original_test)