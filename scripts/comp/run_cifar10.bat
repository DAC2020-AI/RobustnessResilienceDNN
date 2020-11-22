REM scripts\comp\run_cifar10.bat >> logs/cifar10_2.txt 2>>&1

@REM python experiments/quantize/quantize_net.py -m cifar10_shufflenet -lw -ld_name experiments/train/trained_models/cifar10_shufflenetv2.pth -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/cifar10_shufflenet

@REM python experiments/bits/bits.py -c conf -m cifar10_shufflenet -lw -qi 3 -qf 13 -ld_name quantized_weights/cifar10_shufflenet_quantized_3_13 -output cifar10_shufflenet_batch1 -adv cifar10_shufflenet

@REM python experiments/quantize/quantize_net.py -m cifar10_shufflenet -lw -ld_name adversarial-train-custom-loss/cifar10_shufflenetv2_custom_loss.pth -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/cifar10_shufflenet_custom_loss

@REM python experiments/bits/bits.py -c conf -m cifar10_shufflenet -lw -qi 3 -qf 13 -ld_name quantized_weights/cifar10_shufflenet_custom_loss_quantized_3_13 -output cifar10_shufflenet_custom_loss_batch1 -adv cifar10_shufflenet_custom_loss



@REM python experiments/quantize/quantize_net.py -m cifar10_mobilenet -lw -ld_name experiments/train/trained_models/cifar10_mobilenetv2.pth -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/cifar10_mobilenet

@REM python experiments/bits/bits.py -c conf -m cifar10_mobilenet -lw -qi 3 -qf 13 -ld_name quantized_weights/cifar10_mobilenet_quantized_3_13 -output cifar10_mobilenet_batch1 -adv cifar10_mobilenet

@REM python experiments/quantize/quantize_net.py -m cifar10_mobilenet -lw -ld_name adversarial-train-custom-loss/cifar10_mobilenetv2_custom_loss.pth -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/cifar10_mobilenet_custom_loss

@REM python experiments/bits/bits.py -c conf -m cifar10_mobilenet -lw -qi 3 -qf 13 -ld_name quantized_weights/cifar10_mobilenet_custom_loss_quantized_3_13 -output cifar10_mobilenet_custom_loss_batch1 -adv cifar10_mobilenet_custom_loss


python experiments/quantize/quantize_net.py -m cifar10_resnet50 -lw -ld_name experiments/train/trained_models/cifar10_resnet50.pth -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/cifar10_resnet50

python experiments/bits/bits.py -c conf -m cifar10_resnet50 -lw -qi 3 -qf 13 -ld_name quantized_weights/cifar10_resnet50_quantized_3_13 -output cifar10_resnet50_batch1 -adv cifar10_resnet50

python experiments/quantize/quantize_net.py -m cifar10_resnet50 -lw -ld_name adversarial-train-custom-loss/cifar10_resnet50_custom_loss.pth -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/cifar10_resnet50_custom_loss

python experiments/bits/bits.py -c conf -m cifar10_resnet50 -lw -qi 3 -qf 13 -ld_name quantized_weights/cifar10_resnet50_custom_loss_quantized_3_13 -output cifar10_resnet50_custom_loss_batch1 -adv cifar10_resnet50_custom_loss

