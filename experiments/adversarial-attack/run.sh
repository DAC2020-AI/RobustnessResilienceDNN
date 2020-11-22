
mkdir -p logs
mkdir -p logs/mnist_lstm
mkdir -p adversarial

TRAINED_MODELS_DIR="experiments/train/trained_models"

SEED=0
EPSILON=0.3

fname="mnist_lstm_($EPSILON)_$SEED"
THEANO_FLAGS='device=gpu' 

# python3 experiments/adversarial-attack/adversarial-attack.py -c conf -m mnist_lstm -lw -ld_name experiments/train/trained_models/mnist_lstm_10 -ap adversarial/mnist_lstm_100
# python3 experiments/adversarial-attack/adversarial-attack.py -c conf -m mnist_lstm -lw -ld_name $TRAINED_MODELS_DIR/mnist_lstm_100 -e $EPSILON -seed $SEED -ap adversarial/mnist_lstm_100
# python3 experiments/adversarial-attack/adversarial-attack.py -c conf -m mnist_lstm -lw -ld_name adversarial-train-custom-loss/mnist_lstm_100 -e $EPSILON -seed $SEED -ap adversarial/mnist_lstm_100_custom_loss

python3 experiments/adversarial-attack/adversarial-attack.py -c conf -m mnist_lstm -lw -ld_name experiments/train/trained_models/mnist_lstm_errors_2_6_0.0005 -e $EPSILON -seed $SEED -ap adversarial/mnist_lstm_errors_2_6_0.0005
python3 experiments/adversarial-attack/adversarial-attack.py -c conf -m mnist_lstm -lw -ld_name experiments/train/trained_models/mnist_lstm_errors_3_13_0.0005 -e $EPSILON -seed $SEED -ap adversarial/mnist_lstm_errors_3_13_0.0005
