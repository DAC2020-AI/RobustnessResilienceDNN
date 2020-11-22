mkdir -p plots/mnist_lstm_100

# PREFIX="mnist_lstm_errors_2_6_0.0005"
# PREFIX="mnist_lstm_errors_3_13_0.0005"
F1="cifar10_resnet_300_test_accuracy_12_20"
F2="cifar10_resnet_300_test_accuracy_12_20"
displayname1="cifar10_resnet_300_test_accuracy_12_20"
displayname2="cifar10_resnet_300_test_accuracy_12_20"
DIRNAME="cifar10_resnet_300_test_accuracy_12_20"
mkdir -p plots/"$DIRNAME"

# python3 experiments/plot-results/plot-results.py -prefix mnist_lstm_100

python3 experiments/plot-results/plot-results.py -f1 "$F1" -f2 "$F2" -dir "$DIRNAME" -d1 "$displayname1" -d2 "$displayname2"
# python experiments/plot-results/plot-results.py -f1 cifar10_resnet_300_bits_3_13_test_accuracy_3_13 -f2 cifar10_resnet_300_bits_3_13_test_accuracy_3_13 -dir cifar10_resnet_300_bits_3_13_test_accuracy_3_13 -d1 cifar10_resnet_300_bits_3_13_test_accuracy_3_13 -d2 cifar10_resnet_300_bits_3_13_test_accuracy_3_13
