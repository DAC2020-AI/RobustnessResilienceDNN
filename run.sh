# Sample run.sh
# export PYTHONPATH=$(pwd)/:

# mkdir -p cache
# mkdir -p results

# Training MNIST
# python3 run_models.py -m mnist_fc -eps 10 -v  -to -c conf -sw

# Training CiFar
#python /ares/run_models.py -m cifar10_vgg -eps 10 -v  -to -c /ares/conf

#Training SVHN
#python /ares/run_models.py -m imagenet_resnet50 -eps 10 -v  -to -c /ares/conf
TRAINED_MODELS_DIR="experiments/train/trained_models"
# python3 experiments/train/train.py -c conf -m mnist_lstm -eps 40 -v  -to -sw_name  $TRAINED_MODELS_DIR/mnist_lstm_errors_2_6_0.0005 -train_with_errors -frate 0.0005 -qi 2 -qf 6
python3 experiments/train/train.py -c conf -m mnist_lstm -eps 20 -v  -to -sw_name  $TRAINED_MODELS_DIR/mnist_lstm_errors_3_13_0.0005 -train_with_errors -frate 0.0005 -qi 3 -qf 13

SEED=0
# python3 experiments/adversarial-training/adversarial-train.py -c conf -m mnist_lstm -eps 40 -v -seed $SEED -sw_name adversarial-train-custom-loss/mnist_lstm_errors_2_6_0.0005 -train_with_errors -frate 0.0005 -qi 2 -qf 6
python3 experiments/adversarial-training/adversarial-train.py -c conf -m mnist_lstm -eps 20 -v -seed $SEED -sw_name adversarial-train-custom-loss/mnist_lstm_errors_3_13_0.0005 -train_with_errors -frate 0.0005 -qi 3 -qf 13

SEED=0
EPSILON=0.3
# python3 experiments/adversarial-attack/adversarial-attack.py -c conf -m mnist_lstm -lw -ld_name experiments/train/trained_models/mnist_lstm_errors_2_6_0.0005 -e $EPSILON -seed $SEED -ap adversarial/mnist_lstm_errors_2_6_0.0005
python3 experiments/adversarial-attack/adversarial-attack.py -c conf -m mnist_lstm -lw -ld_name experiments/train/trained_models/mnist_lstm_errors_3_13_0.0005 -e $EPSILON -seed $SEED -ap adversarial/mnist_lstm_errors_3_13_0.0005

FRATE=0.00005
QI=2
QF=6
# python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name experiments/train/trained_models/mnist_lstm_errors_2_6_0.0005 -frate $FRATE -seed $SEED -output mnist_lstm_errors_2_6_0.0005
# python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name adversarial-train-custom-loss/mnist_lstm_errors_2_6_0.0005 -frate $FRATE -seed $SEED -output mnist_lstm_errors_2_6_0.0005_adv

QI=3
QF=13
python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name experiments/train/trained_models/mnist_lstm_errors_3_13_0.0005 -frate $FRATE -seed $SEED -output mnist_lstm_errors_3_13_0.0005
python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name adversarial-train-custom-loss/mnist_lstm_errors_3_13_0.0005 -frate $FRATE -seed $SEED -output mnist_lstm_errors_3_13_0.0005_adv
