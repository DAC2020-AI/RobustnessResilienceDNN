'''Models provided for experimentation.'''

# MNIST Models
from .mnist.mnist_fc          import mnistFC
from .mnist.mnist_lstm          import mnistLSTM
from .mnist.mnist_lenet5      import mnistLenet5

# SVHN Models
from .svhn.svhn_lenet5       import svhnLenet5

# Imagenet Models
from .imagenet.imagenet_vgg16    import imagenetVGG16
from .imagenet.imagenet_resnet50 import imagenetResNet50
from .imagenet.imagenet_inceptionv3 import imagenetInceptionv3
# Cifar10 Models
from .cifar10.cifar10_vgg   import cifar10VGG
from .cifar10.cifar10_alexnet   import cifar10alexnet
from .cifar10.cifar10_resnet   import cifar10RESNET
from .cifar10.cifar10_resnet50   import cifar10RESNET50
from .cifar10.cifar10_mobilenetv2   import cifar10MOBILENET
from .cifar10.cifar10_shufflenetv2   import cifar10SHUFFLENET

from .cifar100.cifar100_resnet   import cifar100RESNET
from .cifar100.cifar100_resnet18   import cifar100RESNET18
from .cifar100.cifar100_mobilenetv2   import cifar100MOBILENET
from .cifar100.cifar100_shufflenetv2   import cifar100SHUFFLENET

# TIDigits Models
from .tidigits.tidigits_gru  import tidigitsGRU
from .tidigits.tidigits_lstm import tidigitsLSTM
from .tidigits.tidigits_rnn  import tidigitsRNN

# us_highway
from .us_highway.us_highway_lstm  import usHighWayLSTM

# prediction maintenance
from .pred_main.pred_main_lstm  import predMainLSTM
from .pred_main.pred_main_tcn  import predMainTCN

# seq mnist
from .seq_mnist.seq_mnist_lstm  import seqMnistLSTM

from .base              import ModelBase, IndirectModel, AugmentedDataset
