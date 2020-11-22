
mkdir -p logs
mkdir -p logs/mnist_lstm
mkdir -p adversarial-train
mkdir -p adversarial-train-custom-loss

TRAINED_MODELS_DIR="experiments/train/trained_models"

SEED=0
EPSILON=0.1

fname="mnist_lstm_($EPSILON)_$SEED"
THEANO_FLAGS='device=gpu' 
# python3 experiments/bits/bits.py -c conf -m mnist_fc -lw -qi 2 -qf 6 -ld_name $TRAINED_MODELS_DIR/mnist_fc_quantized_2_6 -frate $FRATE -seed $SEED | tee -a logs/mnist_fc/$fname
# python3 experiments/adversarial-training/adversarial-train.py -c conf -m mnist_lstm -lw -ld_name $TRAINED_MODELS_DIR/mnist_lstm_100 -eps 10 -v -seed $SEED -ap adversarial/mnist_lstm_100 -sw_name adversarial-train/mnist_lstm_100
# python3 experiments/adversarial-training/adversarial-train.py -c conf -m mnist_lstm -eps 100 -v -seed $SEED -sw_name adversarial-train-custom-loss/mnist_lstm_100


python3 experiments/adversarial-training/adversarial-train.py -c conf -m mnist_lstm -eps 40 -v -seed $SEED -sw_name adversarial-train-custom-loss/mnist_lstm_errors_2_6_0.0005 -train_with_errors -frate 0.0005 -qi 2 -qf 6
python3 experiments/adversarial-training/adversarial-train.py -c conf -m mnist_lstm -eps 40 -v -seed $SEED -sw_name adversarial-train-custom-loss/mnist_lstm_errors_3_13_0.0005 -train_with_errors -frate 0.0005 -qi 3 -qf 13
