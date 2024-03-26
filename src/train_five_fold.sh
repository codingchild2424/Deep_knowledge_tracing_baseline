#!/bin/bash
# test

# python train_five_fold.py \
# --model_fn dkt.pth \
# --model_name dkt \
# --dataset_name dkt_dbe22 \
# --batch_size 512 \
# --fivefold True \
# --n_epochs 200

# python train_five_fold.py \
# --model_fn dkvmn.pth \
# --model_name dkvmn \
# --dataset_name dbe22 \
# --batch_size 512 \
# --fivefold True \
# --n_epochs 200

# python train_five_fold.py \
# --model_fn sakt.pth \
# --model_name sakt \
# --dataset_name dkt_dbe22 \
# --batch_size 512 \
# --fivefold True \
# --n_epochs 200

python train_five_fold.py \
--model_fn gkt.pth \
--model_name gkt_pam \
--dataset_name dbe22 \
--batch_size 512 \
--fivefold True \
--n_epochs 200

# python train_five_fold.py \
# --model_fn akt.pth \
# --model_name akt \
# --dataset_name dbe22_pid \
# --batch_size 512 \
# --fivefold True \
# --n_epochs 200

