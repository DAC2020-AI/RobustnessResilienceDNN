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

class mnistLSTMPT(nn.Module):
    def __init__(self):
        super(mnistLSTMPT, self).__init__()
        self.LSTM_feat = 128
        self.seq_size = 28
        self.features  = 28
        self.nb_classes = 10
    
        self.LSTM = nn.LSTM(self.features, self.LSTM_feat)
        self.fc1 = nn.Linear(self.seq_size * self.LSTM_feat, 100)
        self.fc2 = nn.Linear(100, self.nb_classes)

    def forward(self, x):
        batch_size = x.size(0)   

        hidden = [Variable(torch.rand(1, batch_size, self.LSTM_feat, device = x.device)),
                    Variable(torch.rand(1, batch_size, self.LSTM_feat, device = x.device))]

        self.LSTM.flatten_parameters()
        x = x.view(-1, self.seq_size,self.features)
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

class mnistLSTM(ModelBase):
  def __init__(self):
    super(mnistLSTM,self).__init__('mnist','lstm')

    self.param_layer_ids       = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001', '0.001','0.001', '0.001','0.001', '0.001']

    self.lr = .005

  def build_model(self,):
    model = mnistLSTMPT()
    
    self.set_model( model, self.param_layer_ids, self.default_prune_factors )

  def load_dataset(self, ):
    sequence_length = 28
    input_size = 28
    if self.adversarial_path == "":
      transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
      trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
      testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
      
      self.set_data(trainset, testset, testset)
    else:
      transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
      mnist_train = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
      mnist_test = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)  

      adversarial_dataset = pickle.load( open(self.adversarial_path, "rb" ) )

      adv_train = []
      for i in range(len(mnist_train)):
        adv_train.append(mnist_train[i])
      for i in range(len(adversarial_dataset[0])):
        for j in range(len(adversarial_dataset[0][i])):
          adv_train.append((adversarial_dataset[0][i][j],int(adversarial_dataset[1][i][j])))

      adv_test = []
      for i in range(len(mnist_test)):
        adv_test.append(mnist_test[i])
      self.original_test = AugmentedDataset(adv_test[:len(adv_test)])
      for i in range(len(adversarial_dataset[2])):
        for j in range(len(adversarial_dataset[2][i])):
          adv_test.append((adversarial_dataset[2][i][j],int(adversarial_dataset[3][i][j])))
      self.adversarial_test = AugmentedDataset(adv_test[len(self.original_test):])
      
      adv_train_dataset = AugmentedDataset(adv_train)
      adv_test_dataset = AugmentedDataset(adv_test)
      self.set_data(adv_train_dataset, adv_test_dataset, adv_test_dataset)

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