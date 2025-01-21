#!/bin/bash

my_array=(2 3 5 6 8 9 8998 8999)

model_path='output/storage-art'
#model_path='output/fridge-art'

for item in "${my_array[@]}"; do
    CUDA_VISIBLE_DEVICES=3 python render.py -m $model_path -w --skip_train --iteration "$item"
done
