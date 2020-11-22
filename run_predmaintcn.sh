TRAINED_MODELS_DIR="experiments/train/trained_models"
modelname="predMainTCN_100"
modeltype="predMainTCN"
epsilon="0.3"
fname="quantized_weights/predMainTCN_100"
PYTHON="python3"
# PYTHON="python3"
echo "Started: $(date)"
$PYTHON experiments/train/train.py -c conf -m $modeltype -eps 100 -v  -to -sw_name  $TRAINED_MODELS_DIR/"$modelname"

QI=2
QF=6
for QF in 6 8 10 12
do
  $PYTHON experiments/quantize/quantize_net.py -m $modeltype -lw -ld_name $TRAINED_MODELS_DIR/"$modelname" -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname 
done
QI=3
QF=13
$PYTHON experiments/quantize/quantize_net.py -m $modeltype -lw -ld_name $TRAINED_MODELS_DIR/"$modelname" -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname 

$PYTHON experiments/adversarial-attack/adversarial-attack.py -c conf -m $modeltype -lw -ld_name experiments/train/trained_models/"$modelname" -e $epsilon -seed 0 -ap adversarial/"$modelname"_"$epsilon"

QI=2
QF=6
for QF in 6 8 10 12
do
   $PYTHON experiments/bits/bits.py -c conf -m $modeltype -lw -qi $QI -qf $QF -ld_name quantized_weights/"$modelname"_quantized_"$QI"_"$QF" -output "$modelname"_"$epsilon" -adv "$modelname"_"$epsilon"
done
QI=3
QF=13
$PYTHON experiments/bits/bits.py -c conf -m $modeltype -lw -qi $QI -qf $QF -ld_name quantized_weights/"$modelname"_quantized_"$QI"_"$QF" -output "$modelname"_"$epsilon" -adv "$modelname"_"$epsilon"

echo "Original model was processed."

modelname="predMainTCN_100_custom_loss_$epsilon"
advname="predMainTCN_100_$epsilon"
fname="quantized_weights/predMainTCN_100_"$epsilon"_custom_loss"

$PYTHON experiments/adversarial-training/adversarial-train.py -c conf -m $modeltype -eps 100 -v -seed 0 -sw_name adversarial-train-custom-loss/"$modelname" -e $epsilon

QI=2
QF=6
for QF in 6 8 10 12
do
   $PYTHON experiments/quantize/quantize_net.py -m $modeltype -lw -ld_name adversarial-train-custom-loss/"$modelname" -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname 
done
QI=3
QF=13
$PYTHON experiments/quantize/quantize_net.py -m $modeltype -lw -ld_name adversarial-train-custom-loss/"$modelname" -qi $QI -qf $QF --cache cache/ --conf conf --results results -sw $fname 

QI=2
QF=6
for QF in 6 8 10 12
do
   $PYTHON experiments/bits/bits.py -c conf -m $modeltype -lw -qi $QI -qf $QF -ld_name quantized_weights/"$modelname"_quantized_"$QI"_"$QF" -output "$modelname" -adv "$advname"
done
QI=3
QF=13
$PYTHON experiments/bits/bits.py -c conf -m $modeltype -lw -qi $QI -qf $QF -ld_name quantized_weights/"$modelname"_quantized_"$QI"_"$QF" -output "$modelname" -adv "$advname"




