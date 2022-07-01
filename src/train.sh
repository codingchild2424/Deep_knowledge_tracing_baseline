#!/bin/bash

datasets="assist2009 assist2012 assist2015 assist2017 algebra2005 algebra2006"

for dataset2 in ${datasets2}
do
    python \
    train.py \
    --model_fn model.pth \
    --dataset_name ${dataset2} \
    --batch_size 256 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --fivefold True \
    --n_epochs 1000
done