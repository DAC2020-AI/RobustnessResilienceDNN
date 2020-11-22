import dl_models
import argparse
import sys
import cProfile
import hickle
import copy
import random

from dl_models.models.base import *

from dl_models.models import mnistFC
from dl_models.models import mnistLSTM
from dl_models.models import mnistLenet5
from dl_models.models import svhnLenet5
from dl_models.models import imagenetVGG16
from dl_models.models import imagenetResNet50
from dl_models.models import imagenetInceptionv3
from dl_models.models import cifar10VGG
from dl_models.models import tidigitsGRU
from dl_models.models import tidigitsRNN
from dl_models.models import tidigitsLSTM
from dl_models.models import cifar10alexnet
from dl_models.models import predMainLSTM
from dl_models.models import predMainTCN
from dl_models.models import cifar10RESNET

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import pickle

model_class_map = {
                   'mnist_lenet5'      : mnistLenet5,
                   'mnist_fc'          : mnistFC,
                   'mnist_lstm'          : mnistLSTM,
                   'svhn_lenet5'       : svhnLenet5,
                   'imagenet_vgg16'    : imagenetVGG16,
                   'imagenet_resnet50' : imagenetResNet50,
                   'cifar10_vgg'       : cifar10VGG,
                   'tidigits_gru'      : tidigitsGRU,
                   'tidigits_rnn'      : tidigitsRNN,
                   'tidigits_lstm'      : tidigitsLSTM,
                   'imagenet_inceptionv3' : imagenetInceptionv3,
                   'cifar10_alexnet'   : cifar10alexnet,
                   'predMainLSTM'   : predMainLSTM,
                   'predMainTCN'   : predMainTCN,
                   'cifar10_resnet'   : cifar10RESNET,
                  }

class AugmentedDataset(Dataset):
  def __init__(self, combined_list, transform=None):
    self.combined_list = combined_list
    
  def __len__(self):
    return len(self.combined_list)
  
  def __getitem__(self, idx):
    sample = self.combined_list[idx]      
    return sample

def cli():
  # Default message.
  parser = argparse.ArgumentParser(description='Bit level fault injection experiment', \
                                        epilog='Configure your experiment from the command line.')

  parser.add_argument('-m', '--model', required=True, type=str, \
                       help='Pick a model to run. Models listed in models/model_config.py')

  parser.add_argument('-lw', '--load_weights', action='store_true', help='Load saved weights from cache.')

  parser.add_argument('-ld_name', '--weight_name', default=None, type=str, \
                           help='Specifiy the weights to use.')

  parser.add_argument('-seed', '--seed', default=0xdeadbeef, type=int, help='Random seed for bit-level fault injector')
  parser.add_argument('-sw_name', '--sw_name', default=None, type=str, \
                           help='Specify path for saving trained weights')
  parser.add_argument('-l2', '--l2',      default='0.0', type=float, help='Set l2 reg penalty.')
  parser.add_argument('-dr_rt', '--dropout_rate', default='0.35', type=float, help='Dropout rate for appropriate layers.')

  parser.add_argument('-eps'       , '--epochs'    , default=15     , type=int   , help='Num of training epochs.')
  parser.add_argument('-lr'        , '--lr'        , default=0.0001 , type=float , help='Set learning rate for model')
  parser.add_argument('-e'        , '--epsilon'        , default=0.3 , type=float , help='Epsilon for adversarial training')
  parser.add_argument('-v','--verbose', action='store_true', default=False, help='Enable verbose training output')
  parser.add_argument('-ap', '--adversarial_path', default="adversarial/test", type=str, help='Adversarial input path')

  parser.add_argument('-train_with_errors', '--train_with_errors', action='store_true', help='Train the model with errors.')
  parser.add_argument('-frate', '--frate', default=0.0005 , type=float , help='Bit error rate for training')
  parser.add_argument('-qi', '--qi', default=2 , type=int , help='Integer bit for quantization')
  parser.add_argument('-qf', '--qf', default=6 , type=int , help='Fraction bit for quantization')

  parser.add_argument('-c','--configuration', type=str, default=None, help='Specify a configuration file.')
  parser.add_argument('-cache','--cache', type=str, default=None, help='Specify a cache dir.')
  parser.add_argument('-results','--results', type=str, default=None, help='Specify results dir.')

  args = parser.parse_args()

  return args

def load_and_build(model, args):
  # build the model 
  model.load_dataset()
  model.set_training_params(args)

  model.build_model()
  model.compile_model()

  if args.load_weights:
    model.load_weights(args.weight_name, absolute=True)

def exp(model, args):
  load_and_build(model, args)
  
  if not args.train_with_errors:
    model.fit_adv_model(v=args.verbose,epsilon=args.epsilon)
  else:
    q = (args.qi, args.qf)
    frate = args.frate
    print("q: ",q)
    print("frate: ",frate)
    model.fit_adv_model_with_error_injection(q, frate, v=args.verbose)

  model.save_weights(args.sw_name)

  error = model.eval_model()
  print('Validation error after training: %f' % error)
  error = model.test_model()
  print('Test error after training: %f' % error)
  

def config_setup(args):
  if args.configuration is not None:
    print("[Conf] Using configuration from:" + args.configuration)
    Conf.load(Conf.find_config(args.configuration))
  else:
    print("[Conf] Using default environment configuration")
    Conf.set_env_default()

  if args.cache is not None:
    Conf.set('cache', args.cache)

  if args.results is not None:
    Conf.set('results', args.results)

if __name__=='__main__':
  args = cli()
  np.random.seed(args.seed)
  config_setup(args)
  model_name = args.model
  ModelClass = model_class_map[model_name]
  model = ModelClass()
  print('Experimenting with model: %s' % model.model_name)
  exp(model, args)
