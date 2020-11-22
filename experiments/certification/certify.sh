
TRAINED_MODELS_DIR="experiments/train/trained_models"
CERTIFICATION_RESULT_DIR="experiments/certification/results"
CERTIFICATION_PLOT_DIR="experiments/certification/plots"
mkdir -p $CERTIFICATION_RESULT_DIR
mkdir -p $CERTIFICATION_PLOT_DIR


python3 experiments/certification/certify.py $CERTIFICATION_RESULT_DIR/seq_mnist_lstm_quantized_3_13.txt -c conf -m seq_mnist_lstm -v -sigma 0.5 --N0 100 --N 100 \
	-lw -ld_name quantized_weights/seq_mnist_lstm_quantized_2_6 -plot_certified_accuracy 

# For certification
# echo "Started: $(date)"
# python3 experiments/certification/certify.py $CERTIFICATION_RESULT_DIR/cifar10_resnet_quantized_3_13.txt -c conf -m cifar10_resnet -v -sigma 0.5 --N0 100 --N 500 \
# 	-lw -ld_name quantized_weights/cifar10_resnet_quantized_3_13 -plot_certified_accuracy 
# echo "Started: $(date)"
# python3 experiments/certification/certify.py $CERTIFICATION_RESULT_DIR/cifar10_resnet_custom_loss_0.03.txt -c conf -m cifar10_resnet -v -sigma 0.5 --N0 100 --N 500 \
# 	-lw -ld_name adversarial-train-custom-loss/cifar10_resnet_custom_loss_0.03 -plot_certified_accuracy
echo "Started: $(date)"
# quantized_weights/cifar10_resnet_quantized_3_13
# adversarial-train-custom-loss/cifar10_resnet_custom_loss_0.03
# For plotting multiple certification results 
# python3 experiments/certification/analyze.py plotfile dataset_name resultfile1 resultfile2 ... resultfileN 

