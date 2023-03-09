# A simple repo for training vision reasoning tasks

This repo contains code for training some vision reasoning tasks.

# Dataset
I-RAVEN: https://github.com/husheng12345/SRAN

PGM: https://github.com/deepmind/abstract-reasoning-matrices

## Dependency
numpy

pytorch

tqdm

matplotlib

opencv-python 

## Training
`python train.py --model_name {name of model} --root {path to dataset} --fig_type {name of regime}  --dataset {path to save model} --cuda 0`