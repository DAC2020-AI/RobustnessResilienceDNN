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
import dl_models.models.cifar10.mobilenetv2 as mobilenetv2

class cifar10MOBILENET(ModelBase):
  def __init__(self):
    super(cifar10MOBILENET,self).__init__('cifar10','CiFar10MOBILENET')
    self.layer_ids             = []
    self.default_prune_factors = []

    # TODO different lr rates
    self.l2 = 5e-4
    self.lr = 0.1
    self.weights_file_name = "experiments/train/trained_models/cifar10_mobilenet_temp"
    
  def load_dataset(self, ):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
    self.set_data(trainset, testset, testset)

  def compile_model(self, loss='categorical_crossentropy', optimizer='sgd', metrics=None):
    super().compile_model(loss, optimizer, metrics)
    self.train_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60,120,160], gamma=0.2)

  def build_model(self,faults=[]):
    module = mobilenetv2.mobilenetv2()

    self.set_model(module, self.layer_ids, self.default_prune_factors)

    


