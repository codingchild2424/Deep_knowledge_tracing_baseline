#!/bin/bash

datasets="assist2009 assist2017 algebra2005 algebra2006"
pid_datasets="assist2009_pid assist2017_pid algebra2005_pid algebra2006_pid"
models="dkt dkvmn sakt"

for pid_dataset in ${pid_datasets}
do
    python \
    train.py \
    --model_fn model.pth \
    --model_name akt \
    --dataset_name ${pid_dataset} \
    --batch_size 256 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --fivefold True \
    --n_epochs 1000
done

for model in ${models}
do
    for dataset in ${datasets}
    do
        python \
        train.py \
        --model_fn model.pth \
        --model_name ${model} \
        --dataset_name ${dataset} \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --fivefold True \
        --n_epochs 1000
    done
done