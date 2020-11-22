REM run_predtcn.bat > logs/predtcn_log.txt 2>&1

@echo off
set modeltype=predMainTCN
set modelname=predMainTCN
set epsilon=0.3

@echo Train Started: %date% %time%
python experiments/train/train.py -c conf -m %modeltype% -eps 3 -v  -to -sw_name  experiments/train/trained_models/%modelname%
@echo Quan: %date% %time%

python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 6 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 8 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 10 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 12 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%

@echo Adv attack: %date% %time%
python experiments/adversarial-attack/adversarial-attack.py -c conf -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -e %epsilon% -seed 0 -ap adversarial/%modelname%_%epsilon%
REM python experiments/adversarial-attack/adversarial-attack.py -c conf -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -e %epsilon% -seed 0 -ap adversarial/%modelname%_%epsilon% -art -attack_type FastGradientMethod
@echo Bit: %date% %time%

python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 6 -ld_name quantized_weights/%modelname%_quantized_2_6 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%
@echo Started: %date% %time%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 8 -ld_name quantized_weights/%modelname%_quantized_2_8 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 10 -ld_name quantized_weights/%modelname%_quantized_2_10 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 12 -ld_name quantized_weights/%modelname%_quantized_2_12 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 3 -qf 13 -ld_name quantized_weights/%modelname%_quantized_3_13 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%

set modelname=ppredMainTCN_custom_loss
set advname=predMainTCN
set fname=quantized_weights/predMainTCN_custom_loss

python experiments/adversarial-training/adversarial-train.py -c conf -m %modeltype% -eps 4 -v -seed 0 -sw_name adversarial-train-custom-loss/%modelname% -e 0.3

python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 6 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 8 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 10 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 12 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%

python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 6 -ld_name quantized_weights/%modelname%_quantized_2_6 -output %modelname%_%epsilon% -adv %advname%_%epsilon%
@echo Started: %date% %time%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 8 -ld_name quantized_weights/%modelname%_quantized_2_8 -output %modelname%_%epsilon% -adv %advname%_%epsilon%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 10 -ld_name quantized_weights/%modelname%_quantized_2_10 -output %modelname%_%epsilon% -adv %advname%_%epsilon%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 12 -ld_name quantized_weights/%modelname%_quantized_2_12 -output %modelname%_%epsilon% -adv %advname%_%epsilon%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 3 -qf 13 -ld_name quantized_weights/%modelname%_quantized_3_13 -output %modelname%_%epsilon% -adv %advname%_%epsilon%
