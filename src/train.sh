#!/bin/bash

datasets="assist2009 assist2012 assist2015 assist2017 algebra2005 algebra2006"
model_names="dkt dkvmn sakt"

for dataset in ${datasets}
do
    for model_name in ${model_names}
    do
        python \
        train.py \
        --model_fn model.pth \
        --model_name ${model_name} \
        --dataset_name ${dataset} \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --fivefold True \
        --n_epochs 1000
    done
done