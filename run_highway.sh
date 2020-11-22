#!/bin/bash
#
# CompecTA (c) 2018
#
# TORCH job submission script
#
# TODO:
#   - Set name of the job below changing "TORCH" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch torch_submit.sh
#
# -= Resources =-
#
# SBATCH --job-name=highway
# SBATCH --ntasks=5
# SBATCH --gres=gpu:8
# SBATCH --partition=main
# SBATCH --output=%j-torch.out

################################################################################
source /etc/profile.d/z_compecta.sh
echo "source /etc/profile.d/z_compecta.sh"
################################################################################

# MODULES LOAD...
echo "CompecTA Pulsar..."
module load compecta/pulsar

################################################################################

echo ""
echo "============================== ENVIRONMENT VARIABLES ==============================="
env
echo "===================================================================================="
echo ""
echo ""

PULSAR=/raid/apps/compecta/pulsar/pulsar

echo "Running Tensorflow command..."
echo "===================================================================================="
/raid/users/ykoroglu/anaconda3/bin/python3 experiments/train/train.py -c conf -m usHighWayLSTM -eps 100 -v  -to -sw_name  experiments/train/trained_models/usHighWayLSTM_100 > usHighWayLSTM_100_training.txt
RET=$?

echo ""
echo "===================================================================================="
echo "Solver exited with return code: $RET"
exit $RET