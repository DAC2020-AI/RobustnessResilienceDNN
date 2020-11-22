import dl_models
import argparse

import numpy as np

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
  parser = argparse.ArgumentParser(description='Quantization experiment', \
                                        epilog='Configure you experiment from the command line.')

  parser.add_argument('-m', '--model', required=True, type=str, \
                       help='Pick a model to run. Models listed in models/model_config.py')

  parser.add_argument('-lw', '--load_weights', action='store_true', help='Load saved weights from cache.')
  parser.add_argument('-sw', '--save_path', default=None, type=str, help='Path to save weights')

  parser.add_argument('-ld_name', '--weight_name', default=None, type=str, \
                           help='Specifiy the weights to use.')

  parser.add_argument('-qi', '--qi', default=2, type=int, help='Integer bits for quantization')
  parser.add_argument('-qf', '--qf', default=6, type=int, help='Fractional bits for quantization')

  parser.add_argument('-c','--configuration', type=str, default=None, help='Specify a configuration file.')
  parser.add_argument('-ap','--adversarial_path', type=str, default="", help='Adversarial input path.')

  parser.add_argument('-cache','--cache', type=str, default=None, help='Specify a cache dir.')
  parser.add_argument('-results','--results', type=str, default=None, help='Specify results dir.')
  parser.add_argument('-mmc','--mmc_loss', action='store_true', default=False, help='Use mmc loss')

  args = parser.parse_args()

  return args

def load_and_build(model, args):
  # build the model
  if args.adversarial_path != "":
    model.set_adversarial_path(args.adversarial_path)
  
  model.load_dataset()
  if args.mmc_loss:
    model.init_mmc(args.mmc_loss)
  model.build_model()
  model.compile_model()

  if args.load_weights:
    model.load_weights(args.weight_name, absolute=True)
  else:
    model.fit_model()

def quantize_exp(model, mask, args):
  q = (args.qi, args.qf)
  print("=====================================================================")
  print("Quantize experiment args: ")
  print("  mask: ", mask)
  print("  q: ", q)

  errs = [ ]

  pre_err = model.eval_model()
  print("Initial error: ", pre_err)
  quantizer = dl_models.transform.Quantize(mask, q)
  quantizer(model)
  err = model.eval_model() #eval after quantizing
  print("Quantization: ", q, " \t | Error = ", err)
  model.save_weights(args.save_path + '_quantized_' + str(q[0]) + "_" + str(q[1]))


def exp(model, args):
  load_and_build(model, args)
  mask     = [ True for layer in model.get_layers()]

  print("=====================================================================")
  print('(0) Model Topology')
  print()
  for layer in model.get_layers():
    print('  ->',layer[0],':',layer[1].size())
  quantize_exp(model, mask, args)

def config_setup(args):
  if args.configuration is not None:
    Conf.load(Conf.find_config(args.configuration))
  else:
    Conf.set_env_default()

  if args.cache is not None:
    Conf.set('cache', args.cache)

  if args.results is not None:
    Conf.set('results', args.results)

if __name__=='__main__':
  args = cli()
  config_setup(args)
  model_name = args.model
  print("NAME: " + model_name)
  ModelClass = model_class_map[model_name]
  model = ModelClass()
  print('Experimenting with model: %s' % model.model_name)
  exp(model, args)
