import os
import numpy as np
from timeit import default_timer as timer

from dl_models.configuration import Conf
import dl_models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pickle
import copy

class ModelBase(object):
  def __init__(self, dataset='mnist', model_name='abstract_model'):
    super(ModelBase, self).__init__()

    self.dataset = dataset
    self.model_name = model_name
    self.cache_dir = Conf.get('cache_dir')
    cache_file = '%s_%s' % (dataset, model_name)
    self.weights_file_name = os.path.join(self.cache_dir, cache_file)

    self.model = None

    self.traindata = None
    self.testdata = None
    self.valdata = None

    self.num_epochs = 10
    self.l1 = 0.0
    self.l2 = 0.0
    self.dropout_rate = 0.0

    self.train_scheduler = None

    self.layer_ids = []
    self.layer_prune_rates = []
    
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.total_weights = 0
    self.adversarial_path = ""
    self.original_test = []
    self.adversarial_test = []

  def set_adversarial_path(self,adversarial):
    self.adversarial_path = adversarial

  def set_model(self, model, layer_ids, layer_prune_rates):
    self.model = model
    self.layer_ids = layer_ids
    self.layer_prune_rates = layer_prune_rates
    self.total_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)

  def set_device(self, device):
    self.device = device

  def get_device(self):
    return self.device

  def set_training_params(self, args):
    print(args)
    self.num_epochs = args.epochs
    self.l2 = args.l2
    self.dropout_rate = args.dropout_rate

  def set_data(self, train, test, val):
    self.traindata = train
    self.testdata = test
    self.valdata = val

  def compile_model(self, loss='categorical_crossentropy', optimizer='adadelta', metrics=None):
        if loss == 'categorical_crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'binary_crossentropy':
            self.criterion = nn.BCELoss()
        elif loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "mmc_loss":
            self.criterion = self.MMCLoss()

        if optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), weight_decay=self.l2)
        elif optimizer == 'rms':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = self.lr, weight_decay=self.l2)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, weight_decay = self.l2, momentum=0.9, nesterov=True)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.l2)
        if metrics is None:
            self.metrics=['accuracy']
        else:
            self.metrics = metrics
  def calculate_mean_logits(self, class_number,self_D):
    L = class_number
    # D is a hyper parameter
    D = self_D
    mean_logits = np.zeros((L,D))
    mean_logits[0][0] = 1
    for k in range(1,L):
      for j in range(0,k):
        mean_logits[k,j] = -(1/(L-1)+np.dot(mean_logits[k],mean_logits[j]))/mean_logits[j,j]
      mean_logits[k,k] = np.sqrt(np.absolute(1-np.linalg.norm(mean_logits[k])**2))
    return mean_logits

  def MMCLoss(self):
    # TODO check loss works properly
    def loss(outputs, labels):
      # cross entropy
      # labels = labels.squeeze().long()
      # log_prob = -1.0 * F.log_softmax(outputs, 1)
      # print(log_prob)
      # loss = log_prob.gather(1, labels.unsqueeze(1))
      # print(loss)
      # loss = loss.mean()
      # print(loss)
      # exit()
      # return loss
      
      # method 1
      labels = labels.squeeze().long()
      loss = outputs.gather(1, labels.unsqueeze(1))
      loss = loss.mean()
      return -1*loss
      # method 2
      # y_onehot = labels.numpy()
      # y_onehot = (np.arange(2) == y_onehot[:,None]).astype(np.float32)
      # y_onehot = torch.from_numpy(y_onehot).double()
      # return -1*(((outputs @ y_onehot.squeeze().t())).sum())
      # # method 3
      # return -1*(outputs[range(0,outputs.shape[0]),labels.long().squeeze()].mean())
    return loss

  def save_best(self, best):
    # TODO update new_err as np.sum(new_err)
    new_err = self.test_model()
    if new_err < best:
        print("New best model with error %f", new_err)
        # TODO update weight file
        self.save_weights("experiments/train/trained_models/usHighWayLSTM_100_"+str(new_err))
        return new_err
    return best

  def fit_model(self, batch_size=128, v=0, keep_best=False):
    # TODO update for classification and regression
    #best_acc = 1
    best_acc = 1000000
    self.model.train()
    trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=batch_size, shuffle=True, num_workers=1)
    self.model.to(self.device)
    for e in range(self.num_epochs):
        print("Training epoch " + str(e))
        starttime = timer()
        self.model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            # print("Training error: ", self.check_model(self.traindata))
        if keep_best:
            best_acc = self.save_best(best_acc)
        if (e+1)%50==0:
            self.save_weights(self.weights_file_name+"_"+str(e+1))
        new_err = self.test_model()
        print("Training error: ", self.check_model(self.traindata))
        print("Test error: ",new_err)
        endtime = timer()
        print("Time:",endtime-starttime)
        if self.train_scheduler is not None:
            self.train_scheduler.step()
        
  def fit_model_with_error_injection(self, q, frate, batch_size=128, v=0, keep_best=False):
    # TODO update for classification and regression
    #best_acc = 1
    best_acc = 1000000
    
    layer_mask = [ True for layer in self.get_layers()]
    # quantizer
    quantizer = dl_models.transform.Quantize(layer_mask, q)
    # bit error injector
    random_fault_injector = dl_models.transform.RandomFault(layer_mask, 
                                                                frac=frate,
                                                                random_addrs=True,
                                                                fault_type='bit',
                                                                int_bits=q[0],
                                                                frac_bits=q[1])
    
    lambda_ = 1
    self.model.train()
    trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=batch_size, shuffle=True, num_workers=1)
    self.model.to(self.device)
    for e in range(self.num_epochs):
        print("Training epoch " + str(e))
        starttime = timer()
        self.model.train()
        pickled_time = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)

            stime = timer()
            self_copy = pickle.loads(pickle.dumps(self))
            #self_copy = copy.deepcopy(self)
            etime = timer()
            pickled_time += etime-stime
            torch.manual_seed(e)
            quantizer(self_copy)
            random_fault_injector(self_copy)

            outputs_copy = self_copy.model(inputs)

            loss = lambda_ * self.criterion(outputs, labels) + self.criterion(outputs_copy, labels)

            loss.backward()
            self.optimizer.step()
 
        if keep_best:
            best_acc = self.save_best(best_acc)
        new_err = self.test_model()
        endtime = timer()
        print("Time:",endtime-starttime)
        # print("Time:",pickled_time/len(trainloader))

        print("Training error: ", self.check_model(self.traindata))
        print("Test error: ",new_err)
  # adversarial training with custom loss
  def fit_adv_model(self, epsilon=0.3, batch_size=128, v=0, keep_best=False):
    best_acc = 1

    self.model.train()
    trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=batch_size, shuffle=True, num_workers=1)
    self.model.to(self.device)
    for e in range(self.num_epochs):
        print("Training epoch " + str(e))
        self.model.train()
        starttime = timer()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()

            delta = torch.zeros_like(inputs, requires_grad=True)
            outputs = self.model(inputs+delta)
            adv_loss = self.criterion(outputs, labels)
            adv_loss.backward()
            delta = epsilon * delta.grad.detach().sign()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            adv_outputs = self.model(inputs+delta)

            loss = 0.5*self.criterion(outputs, labels)+0.5*self.criterion(adv_outputs, labels)
            loss.backward()
            self.optimizer.step()
        new_err = self.test_model()
        print("Training error: ", self.check_model(self.traindata))
        print("Test error: ",new_err)
        if (e+1)%50==0:
            self.save_weights(self.weights_file_name+"_adv_"+str(e+1))
        endtime = timer()
        print("Time:",endtime-starttime)
        if keep_best:
            best_acc = self.save_best(best_acc)
        if self.train_scheduler is not None:
            self.train_scheduler.step()

  def fit_adv_model_with_error_injection(self, q, frate, epsilon=0.3, batch_size=128, v=0, keep_best=False):
    # TODO update for classification and regression
    #best_acc = 1
    best_acc = 1000000
    
    layer_mask = [ True for layer in self.get_layers()]
    # quantizer
    quantizer = dl_models.transform.Quantize(layer_mask, q)
    # bit error injector
    random_fault_injector = dl_models.transform.RandomFault(layer_mask, 
                                                                frac=frate,
                                                                random_addrs=True,
                                                                fault_type='bit',
                                                                int_bits=q[0],
                                                                frac_bits=q[1])
    
    self.model.train()
    trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=batch_size, shuffle=True, num_workers=1)
    self.model.to(self.device)
    for e in range(self.num_epochs):
        self.model.train()
        print("Training epoch " + str(e))
        starttime = timer()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()

            delta = torch.zeros_like(inputs, requires_grad=True)
            outputs = self.model(inputs+delta)
            adv_loss = self.criterion(outputs, labels)
            adv_loss.backward()
            delta = epsilon * delta.grad.detach().sign()

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            adv_outputs = self.model(inputs+delta)

            self_copy = pickle.loads(pickle.dumps(self))
            quantizer(self_copy)
            torch.manual_seed(e)
            random_fault_injector(self_copy)
            outputs_copy = self_copy.model(inputs)

            loss = self.criterion(outputs, labels)+self.criterion(adv_outputs, labels)+self.criterion(outputs_copy, labels)

            loss.backward()
            self.optimizer.step()
        new_err = self.test_model()
        if keep_best:
            best_acc = self.save_best(best_acc)
        endtime = timer()
        print("Time:",endtime-starttime)
        print("Training error: ", self.check_model(self.traindata))
        print("Test error: ",new_err)

  def fit_model_with_input_gradient_regularization(self, batch_size=128, v=0, keep_best=False, lambda_ = 0.1):
    # TODO update for classification and regression
    #best_acc = 1
    best_acc = 1000000

    self.model.train()
    trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=batch_size, shuffle=True, num_workers=1)
    self.model.to(self.device)
    for e in range(self.num_epochs):
        print("Training epoch " + str(e))
        starttime = timer()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            inputs.requires_grad = True
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            #Allow gradient on input for regularization
            loss = self.criterion(outputs, labels)
            #Find grad on input wrt output
            loss.backward(retain_graph=True)
            reg_loss = torch.norm(inputs.grad)
            inputs.requires_grad = False
            self.optimizer.zero_grad()
            loss = loss +0.1*reg_loss
            loss.backward()
            self.optimizer.step()
            # print("Training error: ", self.check_model(self.traindata))
        if keep_best:
            best_acc = self.save_best(best_acc)
        new_err = self.test_model()
        endtime = timer()
        print("Time:",endtime-starttime)
        print("Training error: ", self.check_model(self.traindata))
        print("Test error: ",new_err)

  def accuracy(self, out, labels):
    if out.shape[1]>1:
      vals, outputs = torch.max(out, dim=1)
      if len(labels.shape)>1:
        labels = labels.squeeze()
      return torch.sum(outputs == labels).item()
    else:
      rounded = torch.round(out)
      if labels.shape!=rounded.shape and len(rounded.shape)==2 and rounded.shape[1]==1:
        rounded = torch.squeeze(rounded)
      return torch.sum(rounded == labels).item()
  
  def RMSELoss(self,out,labels):
    # TODO update save_best
    return torch.sqrt(torch.mean((labels-out)**2,dim=0))

  def check_model(self, dataset): 
    # TODO update save_best
    self.model.eval()
    losses = []
    count = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=512,shuffle=False,num_workers=1)
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            labels = labels.type(torch.FloatTensor).to(self.device)
            count += list(labels.data.size())[0]
            if 'accuracy' in self.metrics:
                loss = self.accuracy(self.model(inputs).data,labels.data)
                losses.append(loss)
            if "mse" in self.metrics:
                loss = self.RMSELoss(self.model(inputs).data,labels.data)
                losses.append(loss.numpy())
    total_loss = 0
    if 'accuracy' in self.metrics:
        total_loss = 1. - np.sum(losses) / count
    if "mse" in self.metrics:
        losses = np.array(losses)
        total_loss = np.sum(losses,axis=0) / losses.shape[0] # count
    return total_loss
  
  def check_model_loss(self, dataset): 
    self.model.eval()
    losses = []
    count = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=512,shuffle=False,num_workers=1)
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            labels = labels.type(torch.FloatTensor).to(self.device)
            loss = self.criterion(self.model(inputs).data, labels.data)
            losses.append(loss)
            count += list(labels.data.size())[0]
    return np.sum(losses) / count

  def eval_model_loss(self):
    return self.check_model_loss(self.valdata)

  def test_model_loss(self):
    return self.check_model_loss(self.testdata)

  def eval_model(self, v=0):
    return self.check_model(self.valdata)
  def test_model(self, v=0):
    return self.check_model(self.testdata)

  #Foward hooks, used to handle activation injections
  def register_hook(self, hook, module_ind):
    list(self.model.modules())[module_ind].register_forward_hook(hook)
  def get_modules(self):
    return self.model.modules()

  def save_weights(self, filename=None):
    if filename:
      fname = filename
    else:
      fname = self.weights_file_name
    print("FILENAME = ", fname)
    torch.save(self.model.state_dict(), fname)

  def load_weights(self, fname=None, absolute=False):
    print("Loading weights at:",fname)
    self.model.load_state_dict(torch.load(fname,map_location=torch.device('cpu')))
    # self.model.load_state_dict(torch.load(fname))
    self.model.to(self.device)

  #Get layers _excluding_ bias layers, (use include_biases flag in transforms to control them)
  def get_layers(self):
    layers = list(self.model.named_parameters())
    nb_layers = []
    for layer in layers:
        if 'bias' not in layer[0]:
            nb_layers.append(layer)
    return nb_layers

  #Internal method used to get all layers, including biases
  def get_all_layers(self): 
    return list(self.model.named_parameters())

  def update_layer(self, layer, new_data):
    layer[1].data = new_data.to(self.device)

class IndirectModel(ModelBase):
  '''A model with indirect indices instead of a normal weight matrix.

  This model cannot be executed directly, as its weights are effectively
  meaningless.
  '''

  def __init__(self, *args, **kwargs):
    super(IndirectModel,self).__init__(*args, **kwargs)
    self.value_table = None

  _NIE_msg = 'Cannot execute this method on an indirect model. Convert to a regular model first.'
  def compile_model(self, *a, **k): raise NotImplementedError(self._NIE_msg)
  def fit_model(self, *a, **k): raise NotImplementedError(self._NIE_msg)
  def eval_model(self, *a, **k): raise NotImplementedError(self._NIE_msg)
  def test_model(self, *a, **k): raise NotImplementedError(self._NIE_msg)
  def save_weights(self, *a, **k): raise NotImplementedError(self._NIE_msg)
  def load_weights(self, *a, **k): raise NotImplementedError(self._NIE_msg)

  def get_values(self):
    '''Return the actual weight values shared by the model.'''
    return self.value_table

class AugmentedDataset(Dataset):
  def __init__(self, combined_list, transform=None):
    self.combined_list = combined_list
    
  def __len__(self):
    return len(self.combined_list)
  
  def __getitem__(self, idx):
    sample = self.combined_list[idx]      
    return sample
