#!/bin/bash

datasets="assist2015 assist2009 assist2017 algebra2005 algebra2006 statics slepemapy"
num_encoders="12 24"
crits="binary_cross_entropy rmse"

for crit in ${crits}
do
    for num_encoder in ${num_encoders}
    do
        for dataset2 in ${datasets2}
        do
            python \
            train.py \
            --model_fn model.pth \
            --dataset_name ${dataset2} \
            --num_encoder ${num_encoder} \
            --crit ${crit} \
            --batch_size 256 \
            --grad_acc True \
            --grad_acc_iter 2 \
            --fivefold True \
            --n_epochs 1000
        done
    done
done