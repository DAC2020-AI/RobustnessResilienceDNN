
mkdir -p logs
mkdir -p logs/cifar10
mkdir -p logs/mnist_lstm
TRAINED_MODELS_DIR="quantized_weights"

SEED=0
FRATE=0.00005

# python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi 2 -qf 6 -ld_name $TRAINED_MODELS_DIR/mnist_lstm_30_adv_quantized_2_6 -frate $FRATE -seed $SEED -ap adversarial/mnist_lstm_30 -output mnist_lstm_30_adv


# QI=2
# QF=6
# for QF in 6 8 10 12
# do
#    python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name $TRAINED_MODELS_DIR/mnist_lstm_100_quantized_"$QI"_"$QF" -frate $FRATE -seed $SEED -output mnist_lstm_100
# done
# QI=3
# QF=13
# python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name $TRAINED_MODELS_DIR/mnist_lstm_100_quantized_"$QI"_"$QF" -frate $FRATE -seed $SEED -output mnist_lstm_100

fname="mnist_quantized_($QI)_($QF)_($FRATE)_$SEED"
THEANO_FLAGS='device=gpu' 
# python3 experiments/bits/bits.py -c conf -m mnist_fc -lw -qi 2 -qf 6 -ld_name $TRAINED_MODELS_DIR/mnist_fc_quantized_2_6 -frate $FRATE -seed $SEED | tee -a logs/mnist_fc/$fname



# QI=2
# QF=6
# for QF in 6 8 10 12
# do
#    python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name $TRAINED_MODELS_DIR/mnist_lstm_100_custom_loss_quantized_"$QI"_"$QF" -frate $FRATE -seed $SEED -output mnist_lstm_100_custom_loss
# done
# QI=3
# QF=13
# python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name $TRAINED_MODELS_DIR/mnist_lstm_100_custom_loss_quantized_"$QI"_"$QF" -frate $FRATE -seed $SEED -output mnist_lstm_100_custom_loss



# python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name $TRAINED_MODELS_DIR/mnist_lstm_30_adv_quantized_"$QI"_"$QF" -frate $FRATE -seed $SEED -ap adversarial/mnist_lstm_30 -output mnist_lstm_30_adv


QI=2
QF=6
# python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name experiments/train/trained_models/mnist_lstm_errors_2_6_0.0005 -frate $FRATE -seed $SEED -output mnist_lstm_errors_2_6_0.0005
python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name adversarial-train-custom-loss/mnist_lstm_errors_2_6_0.0005 -frate $FRATE -seed $SEED -output mnist_lstm_errors_2_6_0.0005_adv

QI=3
QF=13
# python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name experiments/train/trained_models/mnist_lstm_errors_3_13_0.0005 -frate $FRATE -seed $SEED -output mnist_lstm_errors_3_13_0.0005
python3 experiments/bits/bits.py -c conf -m mnist_lstm -lw -qi $QI -qf $QF -ld_name adversarial-train-custom-loss/mnist_lstm_errors_3_13_0.0005 -frate $FRATE -seed $SEED -output mnist_lstm_errors_3_13_0.0005_adv
