#!/bin/bash

my_array=(2 3 2000 4000 8000 15000)
#my_array=(2 3)

#model_path='output/ed-1_01_1-c20-u10k'
#model_path='output/blade_ed-iso1-c40-u10k'
#model_path='output/trained_ed-v2'
model_path='output/blade_trained_v2'

for item in "${my_array[@]}"; do
    CUDA_VISIBLE_DEVICES=6 python render.py -m $model_path -w --skip_train --iteration "$item"
done
