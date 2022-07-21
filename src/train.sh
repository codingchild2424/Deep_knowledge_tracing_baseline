#!/bin/bash
model_names="sakt dkvmn"
datasets="algebra2006"

for dataset in ${datasets}
do
    for model_name in ${model_names}
    do
        python \
        train.py \
        --model_fn sakt_model.pth \
        --model_name ${model_name} \
        --dataset_name ${dataset} \
        --batch_size 512 \
        --fivefold True \
        --crit rmse \
        --n_epochs 1000
    done
done