import dl_models
import argparse
import sys
import cProfile
import hickle
import copy
import random
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


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
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
                   'cifar100_resnet'   : cifar100RESNET,
                   'cifar10_resnet'   : cifar10RESNET,
                   'seq_mnist_lstm'   : seqMnistLSTM,
                  }

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
  parser.add_argument('-e', '--epsilon', default=0.1, type=float, help='Epsilon')
  parser.add_argument('-ap', '--adversarial_path', default="adversarial/test", type=str, help='Adversarial input path')

  parser.add_argument('-c','--configuration', type=str, default=None, help='Specify a configuration file.')
  parser.add_argument('-cache','--cache', type=str, default=None, help='Specify a cache dir.')
  parser.add_argument('-results','--results', type=str, default=None, help='Specify results dir.')
  parser.add_argument('-mmc','--mmc_loss', action='store_true', default=False, help='Use mmc loss')

  args = parser.parse_args()

  return args

def load_and_build(model, args):
  # build the model 
  model.load_dataset()
  if args.mmc_loss:
    model.init_mmc(args.mmc_loss)
  model.build_model()
  model.compile_model()

  if args.load_weights:
    model.load_weights(args.weight_name, absolute=True)


def fgsm(model, X, y, epsilon,criterion):
  """ Construct FGSM adversarial examples on the examples X"""
  delta = torch.zeros_like(X, requires_grad=True)
  loss = criterion(model(X + delta), y)
  loss.backward()
  return epsilon * delta.grad.detach().sign()

# def plot_images(X,y,yp,M,N):
#   f,ax = plt.subplots(M,N, sharex=True, sharey=True, figsize=(N,M*1.3))
#   for i in range(M):
#     for j in range(N):
#       ax[i][j].imshow(1-X[i*N+j][0].cpu().numpy(), cmap="gray")
#       title = ax[i][j].set_title("Pred: {}".format(yp[i*N+j].max(dim=0)[1]))
#       plt.setp(title, color=('g' if yp[i*N+j].max(dim=0)[1] == y[i*N+j] else 'r'))
#       ax[i][j].set_axis_off()
#   plt.tight_layout()
#   plt.show()

def exp(model, args):
  load_and_build(model, args)
  epsilon = args.epsilon
  
  # generate adversarial images
  
  train_loader = DataLoader(model.traindata, batch_size = 100, shuffle=True)
  test_loader = DataLoader(model.testdata, batch_size = 100, shuffle=False)
  
  adversarial_dataset = []
  for loader in [train_loader, test_loader]:
    adversaial_inputs = []
    labels = []
    s = 0
    for data in loader:
      X, y = data[0].to(model.device), data[1].to(model.device)

      delta = fgsm(model.model, X, y, epsilon,model.criterion)
      yp = model.model(X + delta)
      y_real = model.model(X)
      outputs = 0
      outputs_real = 0
      if yp.shape[1]>1:
        vals, outputs = torch.max(yp, dim=1)
        vals_real, outputs_real = torch.max(y_real, dim=1)
      else:
        outputs = torch.squeeze(torch.round(yp))
        outputs_real = torch.squeeze(torch.round(y_real))
      adversaial_inputs.append((X + delta)[outputs != outputs_real])
      labels.append(y[outputs != outputs_real].type(torch.FloatTensor)) # extend
      s = s + len(y[outputs != outputs_real])
    adversarial_dataset.append(adversaial_inputs)
    adversarial_dataset.append(labels)
    print("adversarial samples: ",s)
  outfile = open(args.adversarial_path,'wb')
  pickle.dump(adversarial_dataset,outfile)
  outfile.close()

  # draw examples
  # for X,y in test_loader:
  #   yp = model.model(X)
  #   plot_images(X, y, yp, 3, 6)
  #   delta = fgsm(model.model, X, y, epsilon)
  #   yp = model.model(X + delta)
  #   plot_images(X+delta, y, yp, 3, 6)  
  #   break
  

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
