import operator as op

from dl_models.models.base import *
import sys
from functools import reduce

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import dl_models.models.cifar100.resnet as resnet

class cifar100RESNET18(ModelBase):
  def __init__(self):
    super(cifar100RESNET18,self).__init__('cifar100','CiFar100RESNET18')
    self.layer_ids             = []
    self.default_prune_factors = []

    # TODO different lr rates
    self.l2 = 5e-4
    self.lr = 0.1
    self.weights_file_name = "experiments/train/trained_models/cifar100_resnet18_temp"
    
  def load_dataset(self, ):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transform_test)
    self.set_data(trainset, testset, testset)

  def compile_model(self, loss='categorical_crossentropy', optimizer='sgd', metrics=None):
    super().compile_model(loss, optimizer, metrics)
    self.train_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60,120,160], gamma=0.2)

  def build_model(self,faults=[]):
    module = resnet.resnet18()

    self.set_model(module, self.layer_ids, self.default_prune_factors)

    


