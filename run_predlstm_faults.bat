REM run_predlstm_faults.bat >> logs/predlstm_faults_log.txt 2>>&1

@echo off
set modeltype=predMainLSTM
set modelname=predMainLSTM_3_13_0.0005
set epsilon=0.3

REM @echo Train Started: %date% %time%
REM python experiments/train/train.py -c conf -m %modeltype% -eps 100 -v  -to -sw_name  experiments/train/trained_models/%modelname% -train_with_errors -frate 0.005 -qi 3 -qf 13
REM @echo Quan: %date% %time%

REM python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 6 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
REM python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 8 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
REM python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 10 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
REM python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 2 -qf 12 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
REM python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%

REM @echo Adv attack: %date% %time%
REM python experiments/adversarial-attack/adversarial-attack.py -c conf -m %modeltype% -lw -ld_name experiments/train/trained_models/%modelname% -e %epsilon% -seed 0 -ap adversarial/%modelname%_%epsilon%
REM @echo Bit: %date% %time%

REM python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 6 -ld_name quantized_weights/%modelname%_quantized_2_6 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%
REM @echo Started: %date% %time%
REM python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 8 -ld_name quantized_weights/%modelname%_quantized_2_8 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%
REM python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 10 -ld_name quantized_weights/%modelname%_quantized_2_10 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%
REM python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 12 -ld_name quantized_weights/%modelname%_quantized_2_12 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%
REM python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 3 -qf 13 -ld_name quantized_weights/%modelname%_quantized_3_13 -output %modelname%_%epsilon% -adv %modelname%_%epsilon%
REM @echo Finished: %date% %time%

set modelname=predMainLSTM_3_13_0.0005_custom_loss
set advname=predMainLSTM_3_13_0.0005
set fname=quantized_weights/predMainLSTM_3_13_0.0005_custom_loss

REM @echo Adv Train Started: %date% %time%
REM python experiments/adversarial-training/adversarial-train.py -c conf -m %modeltype% -eps 100 -v -seed 0 -sw_name adversarial-train-custom-loss/%modelname% -e 0.3 -train_with_errors -frate 0.005 -qi 3 -qf 13
REM @echo Finished: %date% %time%

python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name adversarial-train-custom-loss/%modelname% -qi 2 -qf 6 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name adversarial-train-custom-loss/%modelname% -qi 2 -qf 8 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name adversarial-train-custom-loss/%modelname% -qi 2 -qf 10 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name adversarial-train-custom-loss/%modelname% -qi 2 -qf 12 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
python experiments/quantize/quantize_net.py -m %modeltype% -lw -ld_name adversarial-train-custom-loss/%modelname% -qi 3 -qf 13 --cache cache/ --conf conf --results results -sw quantized_weights/%modelname%
@echo Bit Started: %date% %time%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 6 -ld_name quantized_weights/%modelname%_quantized_2_6 -output %modelname%_%epsilon% -adv %advname%_%epsilon%
@echo Bit Finished: %date% %time%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 8 -ld_name quantized_weights/%modelname%_quantized_2_8 -output %modelname%_%epsilon% -adv %advname%_%epsilon%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 10 -ld_name quantized_weights/%modelname%_quantized_2_10 -output %modelname%_%epsilon% -adv %advname%_%epsilon%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 2 -qf 12 -ld_name quantized_weights/%modelname%_quantized_2_12 -output %modelname%_%epsilon% -adv %advname%_%epsilon%
python experiments/bits/bits.py -c conf -m %modeltype% -lw -qi 3 -qf 13 -ld_name quantized_weights/%modelname%_quantized_3_13 -output %modelname%_%epsilon% -adv %advname%_%epsilon%

@echo Finished: %date% %time%
