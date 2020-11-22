import sys
import argparse
from dl_models.models.base import *
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# quantizations = [(2,6),(2,8),(2,10),(2,12),(3,13)]
quantizations = [(2,6),(3,13)]

def cli():
  # Default message.
  parser = argparse.ArgumentParser(description='Plot results', \
                                        epilog='Configure you experiment from the command line.')

  parser.add_argument('-f1', '--filename1', default=None, type=str, help='Filename 1.')
  parser.add_argument('-f2', '--filename2', default=None, type=str, help='Filename 2.')
  parser.add_argument('-d1', '--displayname1', default=None, type=str, help='Displayname 1.')
  parser.add_argument('-d2', '--displayname2', default=None, type=str, help='Displaynam 1.')
  parser.add_argument('-dir', '--directory', default=None, type=str, help='Output directory.')

  args = parser.parse_args()
  return args

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

def createDF(X,Y):
  df = pd.DataFrame({"x":X,"y":Y})
  df.sort_values(by=['x'],inplace=True)
  return df    

def get_filename(filename):
  if "/" in filename:
    return filename[filename.index("/")+1:]
  return filename
def compare_quantization(postfix,filename,directory):
  palette = plt.get_cmap('Set1')
  num=0
  fig = plt.figure()
  ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
  current_filename = filename 
  current_filename += "_" + postfix
  print("Compare quantization: ", current_filename)
  for qi,qf in quantizations:
    num += 1
    quantization = str(qi)+"_"+str(qf)
    current = pickle.load( open("bit-error-results/"+current_filename+"_"+quantization, "rb" ) )
    df = createDF(current[0],current[1])
    ax1.plot(df["x"], df["y"], marker='', color=palette(num), linewidth=1, alpha=0.9, label=quantization)
  
  ax1.set_xlim([5*10**-7, 5*10**-2])
  ax1.set_xscale("log", nonposx='clip')
  plt.legend(loc=2, ncol=2)
  # plt.title("Bit Error Results", loc='left', fontsize=12, fontweight=0, color='orange')
  plt.xlabel("Bit Error Rate")
  plt.ylabel("Error")
  
  # fig.text(.5, .05, "Test errors after bit injection for different quantization of "+ current_filename, ha='center')
  current_filename = get_filename(current_filename)
  plt.savefig('plots/'+directory+"/"+current_filename+".png")

def compare_training(qi,qf,filename1,filename2,directory,displayname1,displayname2):
  palette = plt.get_cmap('Set1')
  fig = plt.figure()
  ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
  quantization = str(qi)+"_"+str(qf)
  
  # current = pickle.load( open("bit-error-results/"+filename1+"_"+quantization, "rb" ) )
  current = pickle.load( open("bit-error-results/"+filename1, "rb" ) )
  # current_acc = pickle.load( open("bit-error-results/predMainTCN_100_accuracy_list/predMainTCN_100_0.3_test_accuracy_list"+"_"+quantization, "rb" ) )
  df = createDF(current[0],current[1])
  # quantization = "2_6"
  # current_adv = pickle.load( open("bit-error-results/"+filename2+"_"+quantization, "rb" ) )
  current_adv = pickle.load( open("bit-error-results/"+filename2, "rb" ) )
  
  # current_adv2 = pickle.load( open("bit-error-results/predMainTCN_100_error_all/predMainTCN_100_error_0.001_0.3_test_accuracy"+"_"+quantization, "rb" ) )
  # current_adv3 = pickle.load( open("bit-error-results/predMainTCN_100_error_all/predMainTCN_100_error_0.0001_0.3_test_accuracy"+"_"+quantization, "rb" ) )
  # df_adv2 = createDF(current_adv2[0],current_adv2[1])
  # df_adv3 = createDF(current_adv3[0],current_adv3[1])
  
  # current_acc = pickle.load( open("bit-error-results/predMainTCN_100_accuracy_list/predMainTCN_100_custom_loss_0.3_test_accuracy_list"+"_"+quantization, "rb" ) )
  # print(current_adv)
  # print(current_adv[0][-6])
  # print(current_adv[1][-6])
  # print(current_acc[0][-6])
  # plt.hist(current_acc[1][-6])
  # plt.show()
  # exit()
  df_adv = createDF(current_adv[0],current_adv[1])
  ax1.plot(df["x"], df["y"], marker='', color=palette(0), linewidth=1, alpha=0.9, label=displayname1)
  ax1.plot(df_adv["x"], df_adv["y"], marker='', color=palette(1), linewidth=1, alpha=0.9, label=displayname2)
  
  # ax1.plot(df_adv2["x"], df_adv2["y"], marker='', color=palette(2), linewidth=1, alpha=0.9, label="0.001")
  # ax1.plot(df_adv3["x"], df_adv3["y"], marker='', color=palette(3), linewidth=1, alpha=0.9, label="0.0001")
  
  ax1.set_xlim([5*10**-7, 5*10**-2])
  ax1.set_xscale("log", nonposx='clip')
  plt.legend(loc=2, ncol=2)
  # plt.title("Bit Error Results", loc='left', fontsize=12, fontweight=0, color='orange')
  plt.xlabel("Bit Error Rate")
  plt.ylabel("Error")
  # fig.text(.5, .05, "Test errors after bit injection"+ \
  #   " for quantization ("+quantization+")", ha='center')
  filename1 = get_filename(filename1)
  filename2 = get_filename(filename2)
  plt.savefig('plots/'+directory+"/"+filename1+"_"+filename2+"_"+quantization+".png")

def compare_training_all(filename1,filename2,directory,displayname1,displayname2):
  print("Compare training: ", filename1,filename2)
  for qi,qf in quantizations:
    compare_training(qi,qf,filename1,filename2,directory,displayname1,displayname2)

def compare_loss(filename,dataset,directory):
  palette = plt.get_cmap('Set1')
  num=0
  fig = plt.figure()
  ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
  current_filename = filename 
  original_loss = pickle.load( open("bit-error-results/"+current_filename+"_original", "rb" ) )
  current_filename += "_"+dataset
  current_filename += "_loss"
  print("Compare loss: ", current_filename)
  for qi,qf in quantizations:
    num += 1
    quantization = str(qi)+"_"+str(qf)
    current = pickle.load( open("bit-error-results/"+current_filename+"_"+quantization, "rb" ) )
    df = createDF(current[0],current[1])
    df["y"] = df["y"] - original_loss
    ax1.plot(df["x"], df["y"], marker='', color=palette(num), linewidth=1, alpha=0.9, label=quantization)
  
  ax1.set_xlim([5*10**-7, 5*10**-2])
  ax1.set_xscale("log", nonposx='clip')
  plt.legend(loc=2, ncol=2)
  plt.title("Bit Error Results", loc='left', fontsize=12, fontweight=0, color='orange')
  plt.xlabel("Bit Error Rate")
  plt.ylabel("Delta Loss")
  
  fig.text(.5, .05, "Delta loss after bit injection for different quantization of "+ current_filename, ha='center')
  plt.savefig('plots/'+directory+"/"+current_filename+".png")

def compare_training_loss(qi,qf,filename1,filename2,dataset,directory):
  palette = plt.get_cmap('Set1')
  fig = plt.figure()
  ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
  quantization = str(qi)+"_"+str(qf)
  # read original
  f1_original_loss = pickle.load( open("bit-error-results/"+filename1+"_original", "rb" ) )
  f2_original_loss = pickle.load( open("bit-error-results/"+filename2+"_original", "rb" ) )
  # add loss to filename
  filename1 = filename1 + "_" + dataset
  filename1 = filename1 + "_loss"
  filename2 = filename2 + "_" + dataset
  filename2 = filename2 + "_loss"
  current = pickle.load( open("bit-error-results/"+filename1+"_"+quantization, "rb" ) )
  df = createDF(current[0],current[1])
  df["y"] = df["y"] - f1_original_loss
  current_adv = pickle.load( open("bit-error-results/"+filename2+"_"+quantization, "rb" ) )
  df_adv = createDF(current_adv[0],current_adv[1])
  df_adv["y"] = df_adv["y"] - f2_original_loss
  ax1.plot(df["x"], df["y"], marker='', color=palette(0), linewidth=1, alpha=0.9, label=filename1)
  ax1.plot(df_adv["x"], df_adv["y"], marker='', color=palette(1), linewidth=1, alpha=0.9, label=filename2)
  ax1.set_xlim([5*10**-7, 5*10**-2])
  ax1.set_xscale("log", nonposx='clip')
  plt.legend(loc=2, ncol=2)
  plt.title("Bit Error Results", loc='left', fontsize=12, fontweight=0, color='orange')
  plt.xlabel("Bit Error Rate")
  plt.ylabel("Delta Loss")
  fig.text(.5, .05, "Delta loss after bit injection"+ \
    " for quantization ("+quantization+")", ha='center')
  filename1 = get_filename(filename1)
  filename2 = get_filename(filename2)
  plt.savefig('plots/'+directory+"/"+filename1+"_"+filename2+"_"+quantization+".png")

def compare_training_loss_all(filename1,filename2,dataset,directory):
  print("Compare training loss: ", filename1,filename2)
  for qi,qf in quantizations:
    compare_training_loss(qi,qf,filename1,filename2,dataset,directory)

if __name__ == '__main__':
  args = cli()
  data = []
  filename1 = args.filename1
  filename2 = args.filename2
  displayname1 = args.displayname1
  displayname2 = args.displayname2
  directory = args.directory

  # before/after adversarial training images 
  # compare_quantization("test_accuracy",filename1,directory) # original network
  # compare_quantization("original_adv_accuracy",filename1,directory) # adv inputs on original network
  # compare_quantization("original_adv_accuracy",filename2,directory) # adv inputs on adv trained network
  # compare_quantization("test_accuracy",filename2,directory) # test inputs on adv trained network

  # before/after adversarial training comparison images
  # compare_training_all(filename1+"_test_accuracy",filename2+"_test_accuracy",directory,displayname1,displayname2)
  compare_training_all(filename1,filename2,directory,displayname1,displayname2)
  # compare_training_all(filename1+"_original_adv_accuracy",filename2+"_original_adv_accuracy",directory,displayname1,displayname2)
    
  # compare_loss(filename1,"test",directory)
  # compare_loss(filename1,"original_adv",directory)
  # compare_loss(filename2,"test",directory)
  # compare_loss(filename2,"original_adv",directory)

  # compare_training_loss_all(filename1,filename2,"test",directory)
  # compare_training_loss_all(filename1,filename2,"original_adv",directory)

  