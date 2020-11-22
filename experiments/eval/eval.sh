
PRETRAINED_WEIGHTS="/ares/quantized_weights/"
# Testing pre-trained weights
# python3 /ares/experiments/eval/eval.py -m mnist_fc -v  -c /ares/conf -lw -ld_name $PRETRAINED_WEIGHTS/mnist_fc_quantized_2_6

HIGHWAY_PATH="experiments/train/trained_models/usHighWayLSTM_100_0.2257535332671493"
python3 experiments/eval/eval.py -m usHighWayLSTM -v  -c conf -lw -ld_name $HIGHWAY_PATH

