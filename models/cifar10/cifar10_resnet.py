import operator as op

from dl_models.models.base import *
import sys
from functools import reduce

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from dl_models.models.cifar10.resnet import *

class cifar10RESNET(ModelBase):
  def __init__(self):
    super(cifar10RESNET,self).__init__('cifar','CiFar10RESNET')
    self.layer_ids             = []
    self.default_prune_factors = []

    # TODO different lr rates
    self.l2 = .1
    self.lr = 0.001
    
  def load_dataset(self, ):
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
    self.set_data(trainset, testset, testset)

  def compile_model(self, loss='categorical_crossentropy', optimizer='adam', metrics=None):
    super().compile_model(loss, optimizer, metrics)

  def build_model(self,faults=[]):
    # TODO different resnet structures
    # ResNet18
    module = ResNet(ResidualBlock, [2, 2, 2])
    self.set_model(module, self.layer_ids, self.default_prune_factors)

    


