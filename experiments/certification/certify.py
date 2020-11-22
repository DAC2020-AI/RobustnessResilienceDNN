# This file is partially from https://github.com/locuslab/smoothing repo
# evaluate a smoothed classifier on a dataset
import os
#import setGPU
from experiments.certification.core import Smooth
from time import time
import torch
import datetime
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import math
import dl_models
import argparse
import sys
import numpy

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
from dl_models.models import seqMnistLSTM
from dl_models.models import cifar10RESNET


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
                   'seq_mnist_lstm'   : seqMnistLSTM,
                  }

def cli():
  # Default message.
  parser = argparse.ArgumentParser(description='Certification experiment', \
                                        epilog='Configure your experiment from the command line.')

  parser.add_argument('-m', '--model', required=True, type=str, \
                       help='Pick a model to run. Models listed in models/model_config.py')

  parser.add_argument('-lw', '--load_weights', action='store_true', help='Load saved weights from cache.')
  parser.add_argument('-ld_name', '--weight_name', default=None, type=str, \
                           help='Specifiy the weights to use.')

  parser.add_argument('-seed', '--seed', default=0xdeadbeef, type=int, help='Random seed')
  parser.add_argument('-eps'       , '--epochs'    , default=15     , type=int   , help='Num of training epochs.')
  parser.add_argument('-l2', '--l2',      default='0.0', type=float, help='Set l2 reg penalty.')
  parser.add_argument('-dr_rt', '--dropout_rate', default='0.35', type=float, help='Dropout rate for appropriate layers.')

  parser.add_argument('-alpha'       , '--alpha'    , default=0.001     , type=float   , help='Probability for returning an incorrect answer')
  parser.add_argument('-sigma'       , '--sigma'    , default=0     , type=float   , help='Standard deviation of gaussian noise')
  parser.add_argument('-v','--verbose', action='store_true', default=False, help='Enable verbose training output')
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument('-plot_certified_accuracy', '--plot_certified_accuracy', action='store_true', help='Plot certified accuracy in different radii.')
  parser.add_argument("outfile", type=str, help="output file")

  parser.add_argument('-c','--configuration', type=str, default=None, help='Specify a configuration file.')
  parser.add_argument('-cache','--cache', type=str, default=None, help='Specify a cache dir.')
  parser.add_argument('-results','--results', type=str, default=None, help='Specify results dir.')
  parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
  parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
  parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
  parser.add_argument("-N0" ,"--N0", type=int, default=100)
  parser.add_argument("-N","--N", type=int, default=100000, help="number of samples to use")
  args = parser.parse_args()

  
  return args

def get_num_classes(model):
  if("predMain" in args.model): 
    num_classes = 2
  else:
    num_classes = len(model.traindata.classes)
  return  num_classes

def load_and_build(model, args):
  # build the model 
  model.load_dataset()
  model.set_training_params(args)

  model.build_model()
  model.compile_model()

  if args.load_weights:
    model.load_weights(args.weight_name, absolute=True)
    print('Testing loaded weights')
    err = model.eval_model()
    print('Validation error before training ', err)

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

class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()

class ApproximateAccuracy(Accuracy):
  def __init__(self, data_file_path: str):
      self.data_file_path = data_file_path

  def at_radii(self, radii: np.ndarray):
      df = pd.read_csv(self.data_file_path, delimiter="\t")
      return np.array([self.at_radius(df, radius) for radius in radii])

  def at_radius(self, df: pd.DataFrame, radius: float):
      return (df["correct"] & (df["radius"] >= radius)).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray):
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))


class Line(object):
  def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
      self.quantity = quantity
      self.legend = legend
      self.plot_fmt = plot_fmt
      self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01):
  radii = np.arange(0, max_radius + radius_step, radius_step)
  plt.figure()
  for line in lines:
      plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

  plt.ylim((0, 1))
  plt.xlim((0, max_radius))
  plt.tick_params(labelsize=14)
  plt.xlabel("radius", fontsize=16)
  plt.ylabel("certified accuracy", fontsize=16)
  plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
  # plt.savefig(outfile + ".pdf")
  plt.tight_layout()
  plt.title(title, fontsize=20)
  plt.tight_layout()
  plt.savefig(outfile + ".png", dpi=300)
  plt.close()

if __name__ == "__main__":
  # load the base classifier
  args = cli()
  np.random.seed(args.seed)
  config_setup(args)
  model_name = args.model
  ModelClass = model_class_map[model_name]
  model = ModelClass()
  print('Experimenting with model: %s' % model.model_name)
  load_and_build(model, args)
  base_classifier = model.model
  # create the smooothed classifier g
  smoothed_classifier = Smooth(base_classifier, get_num_classes(model), args.sigma)
  # prepare output file
  f = open(args.outfile, 'w')
  print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

  # iterate through the dataset
  if(args.split == "test"):
    dataset = model.testdata
  else:
    dataset = model.traindata
  if("predMain" in model_name):
    dataset.X = torch.from_numpy(dataset.X) 
    dataset.y = torch.from_numpy(dataset.y) 
  for i in range(len(dataset)):
    # only certify every args.skip examples, and stop after args.max examples
    if i % args.skip != 0:
      continue
    if i == args.max:
      break
    (x, label) = dataset[i]
    before_time = time()
    # certify the prediction of g around x
    if(torch.cuda.is_available()):
      if type(x)==numpy.ndarray:
        x = torch.from_numpy(x)
      x = x.cuda()
    batch_size = 128
    prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
    after_time = time()
    correct = int(prediction == label)
    time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
    print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
        i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

  f.close()
  if(args.plot_certified_accuracy):
    plot_name = args.outfile.split("/")[-1][:-4]
    plot_certified_accuracy(f"experiments/certification/plots/{plot_name}_{args.sigma}", f"{plot_name}, vary $\sigma$", 1.5, [
      Line(ApproximateAccuracy(args.outfile), f"$\sigma = {args.sigma}$")
      ])
