#!/bin/bash

my_array=(20 200 2000 10000)
model_path='output/ed-1_01_1-c20-u10k'
#model_path='output/blade_ed-iso1-c40-u10k'

# 使用 for 循环遍历数组
for item in "${my_array[@]}"; do
    CUDA_VISIBLE_DEVICES=6 python render.py -m $model_path -w --skip_train --iteration "$item"
done
