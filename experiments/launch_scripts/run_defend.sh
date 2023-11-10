#!/bin/bash

#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='~/Advex-NLP/huggingface'

export model=$1 # llama2 or vicuna
export setup=$2 # behaviors or strings

# Create results folder if it doesn't exist
if [ ! -d "../defends" ]; then
    mkdir "../defends"
    echo "Folder '../defends' created."
else
    echo "Folder '../defends' already exists."
fi

data_offset=50

python3.9 -u ../min_defend.py ../../data/advbench/harmful_${setup}.csv
    # --config="../configs/individual_${model}.py" \
    # --config.attack=gcg \
    # --config.train_data="../../data/advbench/harmful_${setup}.csv" \
    # --config.test_data="../../data/advbench/harmful_${setup}.csv" \
    # --config.result_prefix="../results/individual_${setup}_${model}_gcg_offset${data_offset}" \
    # --config.n_train_data=25 \
    # --config.n_test_data=10 \
    # --config.data_offset=$data_offset \
    # --config.n_steps=1000 \
    # --config.test_steps=100 \