REM scripts\comp\run_seq_mnist_lstm.bat >> logs/seq_mnist_lstm_log.txt 2>>&1

@echo off
set modeltype=seq_mnist_lstm
set modelname=seq_mnist_lstm
set epsilon=0.3
python experiments/train/train.py -c conf -m cifar100_resnet -eps 200 -v  -to -sw_name  experiments/train/trained_models/cifar100_resnet__
REM @echo Train Started: %date% %time%
REM python experiments/train/train.py -c conf -m %modeltype% -eps 100 -v  -to -sw_name  experiments/train/trained_models/%modelname% 
REM @echo Quan: %date% %time%

REM python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 6 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
REM python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%

REM REM @echo Adv attack: %date% %time%
REM REM python experiments/adversarial-attack/adversarial-attack.py -c conf -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -e %epsilon% -seed 0 -ap adversarial/%modelname%_%epsilon%
REM REM @echo Bit: %date% %time%

REM python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 6 -ld_name quantized_weights/%modelname%_quantized_2_6 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%
REM @echo Started: %date% %time%
REM python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 3 -qf 13 -ld_name quantized_weights/%modelname%_quantized_3_13 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%
REM @echo Finished: %date% %time%

set modelname=seq_mnist_lstm_custom_loss
set advname=seq_mnist_lstm
set fname=quantized_weights/seq_mnist_lstm_custom_loss

@echo Adv Train Started: %date% %time%
python experiments/train/train.py -c conf -m %modeltype% -eps 100 -v  -to -sw_name adversarial-train-custom-loss/%modelname% -adv_train -e %epsilon%
@echo Finished: %date% %time%

python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name adversarial-train-custom-loss/%modelname% -qi 2 -qf 6 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name adversarial-train-custom-loss/%modelname% -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
@echo Bit Started: %date% %time%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 6 -ld_name quantized_weights/%modelname%_quantized_2_6 -output %modelname%_%epsilon% -adv %advname%_%epsilon%
python experiments/bits/bits.py -c conf -m cifar10_resnet -lw -qi 3 -qf 13 -ld_name experiments/train/trained_models/cifar10_resnet_300 -output cifar10_resnet_300_bits_3_13 -adv cifar10_resnet_300_0.03

python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 3 -qf 13 -ld_name quantized_weights/%modelname%_quantized_3_13 -output %modelname%_%epsilon% -adv %advname%_%epsilon%

@echo Finished: %date% %time%
