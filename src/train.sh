#!/bin/bash
<<<<<<< HEAD

datasets="assist2009 assist2017 algebra2005 algebra2006"
pid_datasets="assist2009_pid assist2017_pid algebra2005_pid algebra2006_pid"
models="dkt dkvmn sakt"
=======
model_names="sakt dkvmn"
datasets="algebra2006"
>>>>>>> 3a965899191a4ca2ccac25dc3cfd8ddec5de0d7b

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
<<<<<<< HEAD
        --model_fn model.pth \
        --model_name ${model} \
=======
        --model_fn sakt_model.pth \
        --model_name ${model_name} \
>>>>>>> 3a965899191a4ca2ccac25dc3cfd8ddec5de0d7b
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
--dataset_name dkt_assist2009_pid_diff

python \
train.py \
--model_fn model.pth \
--model_name sakt_c_q \
--dataset_name dkt_assist2009_pid

python \
train.py \
--model_fn model.pth \
--model_name sakt_c_rasch \
--dataset_name dkt_assist2009_pid

python \
train.py \
--model_fn model.pth \
--model_name sakt_c_q_diff \
--dataset_name dkt_assist2009_pid_diff
