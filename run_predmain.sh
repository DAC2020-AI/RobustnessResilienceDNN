TRAINED_MODELS_DIR="experiments/train/trained_models"
modelname="predMainLSTM_100"

python3 experiments/train/train.py -c conf -m predMainLSTM -eps 100 -v  -to -sw_name  $TRAINED_MODELS_DIR/"$modelname"

fname="quantized_weights/predMainLSTM_100"
QI=2
QF=6
for QF in 6 8 10 12
do
  python3 experiments/quantize/quantize_net.py -m predMainLSTM -lw -ld_name $TRAINED_MODELS_DIR/"$modelname" -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname 
done
QI=3
QF=13
python3 experiments/quantize/quantize_net.py -m predMainLSTM -lw -ld_name $TRAINED_MODELS_DIR/"$modelname" -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname 

python3 experiments/adversarial-attack/adversarial-attack.py -c conf -m predMainLSTM -lw -ld_name experiments/train/trained_models/"$modelname" -e 0.3 -seed 0 -ap adversarial/"$modelname"

QI=2
QF=6
for QF in 6 8 10 12
do
   python3 experiments/bits/bits.py -c conf -m predMainLSTM -lw -qi $QI -qf $QF -ld_name quantized_weights/"$modelname"_quantized_"$QI"_"$QF" -output "$modelname" -adv "$modelname"
done
QI=3
QF=13
python3 experiments/bits/bits.py -c conf -m predMainLSTM -lw -qi $QI -qf $QF -ld_name quantized_weights/"$modelname"_quantized_"$QI"_"$QF" -output "$modelname" -adv "$modelname"

echo "Original model was processed."

modelname="predMainLSTM_100_custom_loss"
advname="predMainLSTM_100"
# python3 experiments/adversarial-training/adversarial-train.py -c conf -m predMainLSTM -eps 100 -v -seed 0 -sw_name adversarial-train-custom-loss/"$modelname" -e 0.3
fname="quantized_weights/predMainLSTM_100_custom_loss"
QI=2
QF=6
for QF in 6 8 10 12
do
   python3 experiments/quantize/quantize_net.py -m predMainLSTM -lw -ld_name adversarial-train-custom-loss/"$modelname" -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname 
done
QI=3
QF=13
python3 experiments/quantize/quantize_net.py -m predMainLSTM -lw -ld_name adversarial-train-custom-loss/"$modelname" -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname 

QI=2
QF=6
for QF in 6 8 10 12
do
   python3 experiments/bits/bits.py -c conf -m predMainLSTM -lw -qi $QI -qf $QF -ld_name quantized_weights/"$modelname"_quantized_"$QI"_"$QF" -output "$modelname" -adv "$advname"
done
QI=3
QF=13
python3 experiments/bits/bits.py -c conf -m predMainLSTM -lw -qi $QI -qf $QF -ld_name quantized_weights/"$modelname"_quantized_"$QI"_"$QF" -output "$modelname" -adv "$advname"




