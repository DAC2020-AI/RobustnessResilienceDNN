#!/usr/bin/env python


import sys
import argparse

import dl_models

from dl_models.transform import SummarizeSparsity
from dl_models.transform import Retraining

from dl_models.models.base import *

from dl_models.models import mnistFC
from dl_models.models import mnistLSTM
from dl_models.models import mnistLenet5
from dl_models.models import svhnLenet5
from dl_models.models import imagenetVGG16
from dl_models.models import imagenetResNet50
from dl_models.models import imagenetInceptionv3
from dl_models.models import cifar10VGG
from dl_models.models import cifar10RESNET
from dl_models.models import tidigitsGRU
from dl_models.models import tidigitsRNN
from dl_models.models import tidigitsLSTM
from dl_models.models import cifar10alexnet
from dl_models.models import usHighWayLSTM
from dl_models.models import predMainLSTM
from dl_models.models import predMainTCN
from dl_models.models import seqMnistLSTM
from dl_models.models import cifar100RESNET
from dl_models.models import cifar100RESNET18
from dl_models.models import cifar100MOBILENET
from dl_models.models import cifar100SHUFFLENET
from dl_models.models import cifar10RESNET50
from dl_models.models import cifar10MOBILENET
from dl_models.models import cifar10SHUFFLENET

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
                   'usHighWayLSTM'   : usHighWayLSTM,
                   'predMainLSTM'   : predMainLSTM,
                   'predMainTCN'   : predMainTCN,
                   'cifar10_resnet'   : cifar10RESNET,
                   'cifar100_resnet'   : cifar100RESNET,
                   'cifar100_resnet18'   : cifar100RESNET18,
                   'cifar100_shufflenet'   : cifar100SHUFFLENET,
                   'cifar100_mobilenet'   : cifar100MOBILENET,
                   'seq_mnist_lstm'   : seqMnistLSTM,
                   'cifar10_resnet50'   : cifar10RESNET50,
                   'cifar10_shufflenet'   : cifar10SHUFFLENET,
                   'cifar10_mobilenet'   : cifar10MOBILENET,
                  }

def cli():
  # Default message.
  parser = argparse.ArgumentParser(description='Harvard VLSI-ARCH deep learning model framework', \
                                        epilog='Configure you experiment from the command line.')

  parser.add_argument('-m', '--model', required=True, type=str, \
                           help='Pick a model to run. Models listed in models/model_config.py')

  parser.add_argument('-to', '--train_once', action='store_true', help='Train the model.')
  parser.add_argument('-train_with_input_gradient_regularization', '--train_with_input_gradient_regularization', action='store_true', help='Train the model with input gradient regularization.')
  parser.add_argument('-train_with_errors', '--train_with_errors', action='store_true', help='Train the model with errors.')
  parser.add_argument('-frate', '--frate', default=0.0005 , type=float , help='Bit error rate for training')
  parser.add_argument('-qi', '--qi', default=2 , type=int , help='Integer bit for quantization')
  parser.add_argument('-qf', '--qf', default=6 , type=int , help='Fraction bit for quantization')

  parser.add_argument('-sw', '--save_weights', action='store_true', help='Save weights to cache.')
  parser.add_argument('-lw', '--load_weights', action='store_true', help='Load saved weights from cache.')
  parser.add_argument('-ld_name', '--weight_name', default=None, type=str, \
                           help='Specifiy the weights to use.')

  parser.add_argument('-sw_name', '--sw_name', default=None, type=str, \
                           help='Specify path for saving trained weights')

  parser.add_argument('-l2', '--l2',      default='0.0', type=float, help='Set l2 reg penalty.')
  parser.add_argument('-dr_rt', '--dropout_rate', default='0.35', type=float, help='Dropout rate for appropriate layers.')

  parser.add_argument('-eps'       , '--epochs'    , default=15     , type=int   , help='Num of training epochs.')
  parser.add_argument('-lr'        , '--lr'        , default=0.0001 , type=float , help='Set learning rate for model')

  parser.add_argument('-pwdist', '--plot_weight_distributions', action='store_true', help='Plot each layer weight distributions.')
  parser.add_argument('-c','--configuration', type=str, default=None, help='Specify a configuration file.')
  parser.add_argument('-v','--verbose', action='store_true', default=False, help='Enable verbose training output')
  
  parser.add_argument('-mmc','--mmc_loss', action='store_true', default=False, help='Use mmc loss')
  
  parser.add_argument('-adv_train','--adversarial_training', action='store_true', default=False, help='Adversarial training')
  parser.add_argument('-e'        , '--epsilon'        , default=0.3 , type=float , help='Epsilon for adversarial training')

  parser.add_argument('-cache','--cache', type=str, default=None, help='Specify a cache dir.')
  parser.add_argument('-results','--results', type=str, default=None, help='Specify results dir.')
  parser.add_argument('-seed', '--seed', default=0xdeadbeef, type=int, help='Random seed for reproducible training runs')

  args = parser.parse_args()
  return args

def load_and_build(model, args):
  # build the model
  model.load_dataset()
  model.set_training_params(args)
  
  if args.mmc_loss:
    model.init_mmc(args.mmc_loss)
  
  model.build_model()
  model.compile_model()

  if args.load_weights:
    model.load_weights(args.weight_name)
    print('Testing loaded weights')
    err = model.eval_model()
    print('Validation error before training ', err)


def train_and_save(model, args):
  # if we didn't load weights, we can train a new DNN.
  if not args.adversarial_training:
    if not args.train_with_errors:
      if not args.train_with_input_gradient_regularization:
        model.fit_model(v=args.verbose)
      else:
        model.fit_model_with_input_gradient_regularization(v=args.verbose)
    else:
      q = (args.qi, args.qf)
      frate = args.frate
      print("q: ",q)
      print("frate: ",frate)
      model.fit_model_with_error_injection(q, frate,v=args.verbose)
  else:
    if not args.train_with_errors:
      model.fit_adv_model(v=args.verbose,epsilon=args.epsilon)
    else:
      q = (args.qi, args.qf)
      frate = args.frate
      print("q: ",q)
      print("frate: ",frate)
      model.fit_adv_model_with_error_injection(q, frate, v=args.verbose)
  model.eval_model()

  model.save_weights(args.sw_name)

  if args.plot_weight_distributions:
    model.plot_distributions()

  error = model.eval_model()
  print(error)
  print('Validation error after training: %f' % error)
  error = model.test_model()
  print('Test error after training: %f' % error)

def process_model(model, args):
  # Get weights, setup model and build functions
  load_and_build(model, args)

  # train and sweep if necessary
  train_and_save(model, args)

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

if __name__ == '__main__':
  args = cli()
  np.random.seed(args.seed)
  config_setup(args)
  model_name = args.model
  ModelClass = model_class_map[model_name]
  model = ModelClass()
  print('Training model: %s' % model.model_name)
  process_model(model, args)
