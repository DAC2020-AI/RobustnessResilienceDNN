import dl_models
import argparse
import sys
import cProfile
import hickle
import copy
import random
import pickle

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

import copy

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
  parser = argparse.ArgumentParser(description='Bit level fault injection experiment', \
                                        epilog='Configure your experiment from the command line.')

  parser.add_argument('-m', '--model', required=True, type=str, \
                       help='Pick a model to run. Models listed in models/model_config.py')

  parser.add_argument('-lw', '--load_weights', action='store_true', help='Load saved weights from cache.')

  parser.add_argument('-ld_name', '--weight_name', default=None, type=str, \
                           help='Specifiy the weights to use.')

  parser.add_argument('-qi', '--qi', default=2, type=int, help='Integer bits for quantization')
  parser.add_argument('-qf', '--qf', default=6, type=int, help='Fractional bits for quantization')

  parser.add_argument('-seed', '--seed', default=0xdeadbeef, type=int, help='Random seed for bit-level fault injector')
  parser.add_argument('-frate', '--frate', default=0.0001, type=float, help='Fault Rate')
  # TODO remove 
  parser.add_argument('-ap','--adversarial_path', type=str, default="", help='Adversarial input path.')
  parser.add_argument('-output','--output', type=str, default="", help='Output input path.')
  parser.add_argument('-adv','--adversarial_input', type=str, default="", help='Adversarial input path.')

  parser.add_argument('-c','--configuration', type=str, default=None, help='Specify a configuration file.')
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

def load_adversarial_dataset(data_path):
  # load test dataset
  adversarial_dataset = pickle.load( open(data_path, "rb" ) )
  adv_test = []
  for i in range(len(adversarial_dataset[2])):
    for j in range(len(adversarial_dataset[2][i])):
      adv_test.append((adversarial_dataset[2][i][j],int(adversarial_dataset[3][i][j])))
  adv_test_dataset = AugmentedDataset(adv_test)
  return adv_test_dataset

def append_results(results,err_sum,frate):
  for data_type in ["test", "original_adv"]:
    for metric_type in ["accuracy","loss","accuracy_list"]:
      results[data_type][metric_type][0].append(frate)
      results[data_type][metric_type][1].append(err_sum[data_type][metric_type])

def save_results(results,path_prefix,quantization):
  for data_type in ["test", "original_adv"]:
    for metric_type in ["accuracy","loss","accuracy_list"]:
      outfile = open(path_prefix+"_"+data_type+"_"+metric_type+"_"+quantization,'wb')
      pickle.dump(results[data_type][metric_type],outfile)
      outfile.close()
  
      
def exp(model, args):
  # load_and_build(model, args)

  # mask     = [ True for layer in model.get_layers()]

  #retrain  = dl_models.transform.Retraining(layer_mask=mask)
  # sparsity = dl_models.transform.SummarizeSparsity(mode='both')
  # distrib  = dl_models.transform.SummarizeDistribution()

  #retrain.config(model)

  # print("=====================================================================")
  # print('(0) Model Topology')
  # print()
  # for layer in model.get_layers():
  #   print('  ->',layer[0],':',layer[1].size())

  # print('(1) Base model')
  # print()
  # err = model.eval_model()
  # print('(1) error:',err)
  # print()
  # sparsity(model)
  # print('(1) sparsity:',sparsity.get_summary())
  # print()
  # distrib(model)
  # print('(1) distribution:',distrib.get_summary())
  # print()
  results = {
    "test": {
      "accuracy": [[],[]],
      "loss": [[],[]],
      "accuracy_list": [[],[]]
    },
    "original_adv": {
      "accuracy": [[],[]],
      "loss": [[],[]],
      "accuracy_list": [[],[]]
    },
    "custom_adv": {
      "accuracy": [[],[]],
      "loss": [[],[]],
      "accuracy_list": [[],[]]
    }
  }
  load_and_build(model, args)
  adv_path = args.adversarial_input
  # original_adv_data = load_adversarial_dataset("adversarial/"+adv_path)
  # custom_adv_data = load_adversarial_dataset("adversarial/mnist_lstm_100_custom_loss")
  repeation_number = 50
  # save original model loss

  # delta_loss = 0
  # if args.adversarial_path != "":
  #   delta_loss = model.test_loss()
  # else:
  #   delta_loss = model.test_model_loss()
  # outfile = open("bit-error-results/"+args.output+"_original",'wb')
  # pickle.dump(delta_loss,outfile)
  # outfile.close()
  # if args.adversarial_path != "":
  #   delta_loss_robust = model.robust_loss()
  #   outfile = open("bit-error-results/"+args.output+"_original_robust",'wb')
  #   pickle.dump(delta_loss_robust,outfile)
  #   outfile.close()
  
  for i in range(4):
    for j in range(9):
      frate = (9-j)*(10**(-6+i))
      print("Started: ",frate)
      starttime = timer()
      err_sum = {
        "test": {
          "accuracy": 0,
          "loss": 0,
          "accuracy_list": []
        },
        "original_adv": {
          "accuracy": 0,
          "loss": 0,
          "accuracy_list": []
        },
        "custom_adv": {
          "accuracy": 0,
          "loss": 0,
          "accuracy_list": []
        }
      }
      for r in range(repeation_number):
        frate = (9-j)*(10**(-6+i))
        #new_mdl = pickle.loads(pickle.dumps(model))
        new_mdl = copy.deepcopy(model)
        layer_mask = [ True for layer in model.get_layers()]

        random_fault_injector = dl_models.transform.RandomFault(layer_mask, seed=args.seed,
                                                                frac=frate,
                                                                random_addrs=True,
                                                                fault_type='bit',
                                                                int_bits=args.qi,
                                                                frac_bits=args.qf)

        random_fault_injector(new_mdl)
        # save each fault injected model
        # new_mdl.save_weights("bit-error-models/"+args.output+"_"+str(args.qi)+"_"+str(args.qf)+"_"+str(9-j)+"_"+str(-6+i)+"_"+str(r))
        
        err_sum["test"]["accuracy_list"].append(new_mdl.eval_model())
        err_sum["test"]["accuracy"] += err_sum["test"]["accuracy_list"][-1]
        #err_sum["test"]["loss"] += new_mdl.test_model_loss()
        # TODO check cuda error
        # err_sum["original_adv"]["accuracy"] += new_mdl.check_model(original_adv_data)
        # err_sum["original_adv"]["loss"] += new_mdl.check_model_loss(original_adv_data)
        # err_sum["custom_adv"]["accuracy"] += new_mdl.check_model(custom_adv_data)
        # err_sum["custom_adv"]["loss"] += new_mdl.check_model_loss(custom_adv_data)        
      err_sum["test"]["accuracy"] /= repeation_number
      err_sum["test"]["loss"] /= repeation_number
      err_sum["original_adv"]["accuracy"] /= repeation_number
      err_sum["original_adv"]["loss"] /= repeation_number
      # err_sum["custom_adv"]["accuracy"] /= repeation_number
      # err_sum["custom_adv"]["loss"] /= repeation_number
      # add to results
      append_results(results,err_sum,frate)
      endtime = timer()
      print("Time:",endtime-starttime)
  save_results(results,"bit-error-results/"+args.output,str(args.qi)+"_"+str(args.qf))    
  # dargs = vars(args)
  # for key in list(dargs.keys()):
  #   print("::::", key, dargs[key])
  # print("::::", "error", err)

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
