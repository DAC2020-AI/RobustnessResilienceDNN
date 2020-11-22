REM scripts\comp\run_cifar100_shufflenet.bat >> logs/cifar100_shufflenet.txt 2>>&1

python experiments/quantize/quantize_net.py -m cifar100_shufflenet -lw -ld_name experiments/train/trained_models/cifar100_shufflenet.pth -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/cifar100_shufflenet

python experiments/bits/bits.py -c conf -m cifar100_shufflenet -lw -qi 3 -qf 13 -ld_name quantized_weights/cifar100_shufflenet_quantized_3_13 -output cifar100_shufflenet_batch1 -adv cifar100_shufflenet

python experiments/quantize/quantize_net.py -m cifar100_shufflenet -lw -ld_name adversarial-train-custom-loss/cifar100_shufflenet_custom_loss.pth -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/cifar100_shufflenet_custom_loss

python experiments/bits/bits.py -c conf -m cifar100_shufflenet -lw -qi 3 -qf 13 -ld_name quantized_weights/cifar100_shufflenet_custom_loss_quantized_3_13 -output cifar100_shufflenet_custom_loss_batch1 -adv cifar100_shufflenet_custom_loss
