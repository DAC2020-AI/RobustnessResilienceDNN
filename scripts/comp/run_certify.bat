REM scripts\comp\run_certify.bat >> logs/certify_log.txt 2>>&1

@echo off
set modeltype=seq_mnist_lstm
set modelname=seq_mnist_lstm

set TRAINED_MODELS_DIR=experiments/train/trained_models
set CERTIFICATION_RESULT_DIR="experiments/certification/results"
set CERTIFICATION_PLOT_DIR="experiments/certification/plots"

@echo Started: %date% %time%
python experiments/certification/certify.py %CERTIFICATION_RESULT_DIR%/cifar10_resnet_quantized_3_13.txt -c conf -m cifar10_resnet -v -sigma 0.25 --N0 100 --N 1000 -lw -ld_name quantized_weights/cifar10_resnet_quantized_3_13 -plot_certified_accuracy 
@echo Started: %date% %time%
python experiments/certification/certify.py %CERTIFICATION_RESULT_DIR%/cifar10_resnet_custom_loss_0.03.txt -c conf -m cifar10_resnet -v -sigma 0.25 --N0 100 --N 1000 -lw -ld_name quantized_weights/cifar10_resnet_custom_loss_0.03_quantized_3_13 -plot_certified_accuracy
@echo Started: %date% %time%

REM @echo Started: %date% %time%
REM python experiments/certification/certify.py %CERTIFICATION_RESULT_DIR%/seq_mnist_lstm_quantized_3_13.txt -c conf -m seq_mnist_lstm -v -sigma 0.5 --N0 100 --N 1000 -lw -ld_name quantized_weights/seq_mnist_lstm_quantized_3_13 -plot_certified_accuracy 
REM @echo Started: %date% %time%
REM python experiments/certification/certify.py %CERTIFICATION_RESULT_DIR%/seq_mnist_lstm_custom_loss_quantized_3_13.txt -c conf -m seq_mnist_lstm -v -sigma 0.5 --N0 100 --N 1000 -lw -ld_name quantized_weights/seq_mnist_lstm_custom_loss_quantized_3_13 -plot_certified_accuracy
REM @echo Started: %date% %time%