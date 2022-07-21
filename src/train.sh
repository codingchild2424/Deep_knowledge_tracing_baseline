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

python \
train.py \
--model_fn model.pth \
--model_name dkt_c_q \
--dataset_name dkt_assist2009_pid

python \
train.py \
--model_fn model.pth \
--model_name dkt_c_rasch \
--dataset_name dkt_assist2009_pid

python \
train.py \
--model_fn model.pth \
--model_name dkt_c_q_ctt \
--dataset_name dkt_assist2009_pid_diff

python \
train.py \
--model_fn model.pth \
--model_name dkvmn_c_q \
--dataset_name assist2009_pid

python \
train.py \
--model_fn model.pth \
--model_name dkvmn_c_rasch \
--dataset_name assist2009_pid

python \
train.py \
--model_fn model.pth \
--model_name dkvmn_c_q_ctt \
--dataset_name dkt_assist2009_pid_diff \
