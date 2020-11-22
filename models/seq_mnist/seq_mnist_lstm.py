import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import operator as op

from dl_models.models.base import *
import h5py
import sys
import numpy as np
import pickle

class seqMnistLSTMPT(nn.Module):
    def __init__(self):
        super(seqMnistLSTMPT, self).__init__()
        self.LSTM_feat = 128
        self.seq_size = 120
        self.features  = 4
        self.nb_classes = 10
    
        self.LSTM = nn.LSTM(self.features, self.LSTM_feat)
        self.fc1 = nn.Linear(self.seq_size * self.LSTM_feat, 100)
        self.fc2 = nn.Linear(100, self.nb_classes)

    def forward(self, x):
        batch_size = x.size(0)   
        
        hidden = [Variable(torch.rand(1, batch_size, self.LSTM_feat, device = x.device)),
                    Variable(torch.rand(1, batch_size, self.LSTM_feat, device = x.device))]

        self.LSTM.flatten_parameters()
        x = x.view(-1, self.seq_size,self.features).type(torch.FloatTensor).to(x.device)
        x = x.permute([1,0,2])      #Move to seq-batch-samp
        output = []
        for i in range(self.seq_size):   #Loop over TIME dimension to allow hidden layer injects
            out, hidden = self.LSTM(x[i].unsqueeze(0), hidden)
            output.append(out.squeeze(0))
        out = torch.stack(output)
        out = out.permute([1,0,2])  #Return to batch-seq-samp
        out = out.contiguous().view(batch_size, -1)
        out = F.relu(self.fc1(out))
        return F.log_softmax(self.fc2(out), -1)

class seqMnistLSTM(ModelBase):
  def __init__(self):
    super(seqMnistLSTM,self).__init__('seqMnist','lstm')

    self.param_layer_ids       = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001', '0.001','0.001', '0.001','0.001', '0.001']

    self.lr = .005

  def build_model(self,):
    model = seqMnistLSTMPT()
    
    self.set_model( model, self.param_layer_ids, self.default_prune_factors )

  def load_dataset(self, ):
      train_path = "data/seq_mnist/training_dataset"
      test_path = "data/seq_mnist/test_dataset"
      train_dataset = pickle.load( open(train_path, "rb" ) )
      test_dataset = pickle.load( open(test_path, "rb" ) )
      train_dataset.classes = range(10)
      test_dataset.classes = range(10)
      self.set_data(train_dataset, test_dataset, test_dataset)

  def compile_model(self, loss='categorical_crossentropy', optimizer='sgd', metrics=None):
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