TRAINED_MODELS_DIR="experiments/train/trained_models"

mkdir -p $TRAINED_MODELS_DIR

# Training Model
python3 experiments/train/train.py -c conf -m mnist_lstm -eps 100 -v  -to -sw_name  experiments/train/trained_models/mnist_lstm -lw -ld_name experiments/train/trained_models/mnist_lstm
# python3 experiments/train/train.py -c conf -m mnist_fc -eps 1 -v  -to -sw_name  $TRAINED_MODELS_DIR/mnist_fc
# python3 experiments/train/train.py -c conf -m mnist_lstm -eps 100 -v  -to -sw_name  $TRAINED_MODELS_DIR/mnist_lstm_100

# python3 experiments/train/train.py -c conf -m usHighWayLSTM -eps 100 -v  -to -sw_name  $TRAINED_MODELS_DIR/usHighWayLSTM_100

# python3 experiments/train/train.py -c conf -m predMainLSTM -eps 100 -v  -to -sw_name  $TRAINED_MODELS_DIR/predMainLSTM_100

# train with errors
# python3 experiments/train/train.py -c conf -m mnist_lstm -eps 100 -v  -to -sw_name  $TRAINED_MODELS_DIR/mnist_lstm_errors_3_13_0.0005 -train_with_errors -frate 0.0005 -qi 3 -qf 13

python3 experiments/train/train.py -c conf -m predMainTCN -eps 2 -v  -to -sw_name  $TRAINED_MODELS_DIR/predMainTCN_2

python3 experiments/train/train.py -c conf -m cifar10_resnet -eps 100 -v  -to -sw_name  experiments/train/trained_models/cifar10_resnet