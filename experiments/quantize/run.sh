#!/bin/bash
mkdir -p quantized_weights

TRAINED_MODELS_DIR="experiments/train/trained_models"
ADV_TRAINED_MODELS_DIR="adversarial-train-custom-loss"

THEANO_FLAGS='device=gpu' 

QI=3
QF=13

fname="quantized_weights/mnist_lstm_100"

# python3 experiments/quantize/quantize_net.py -m mnist_fc -lw -ld_name $TRAINED_MODELS_DIR/mnist_fc -qi 2 -qf 6 --cache cache/ --conf conf --results results -sw quantized_weights/mnist_fc
# python3 experiments/quantize/quantize_net.py -m mnist_lstm -lw -ld_name $TRAINED_MODELS_DIR/mnist_lstm -qi 2 -qf 6 --cache cache/ --conf conf --results results -sw quantized_weights/mnist_lstm
# python3 experiments/quantize/quantize_net.py -m mnist_lstm -lw -ld_name $TRAINED_MODELS_DIR/mnist_lstm_100 -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname
# python3 experiments/quantize/quantize_net.py -m mnist_lstm -lw -ld_name $ADV_TRAINED_MODELS_DIR/mnist_lstm_100 -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname -ap adversarial/mnist_lstm_100


fname="quantized_weights/mnist_lstm_100_custom_loss"
QI=2
QF=6
for QF in 6 8 10 12
do
   python3 experiments/quantize/quantize_net.py -m mnist_lstm -lw -ld_name $ADV_TRAINED_MODELS_DIR/mnist_lstm_100 -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname 
done
QI=3
QF=13
python3 experiments/quantize/quantize_net.py -m mnist_lstm -lw -ld_name $ADV_TRAINED_MODELS_DIR/mnist_lstm_100 -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname 
