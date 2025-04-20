#!/bin/bash

my_array=(9 19 20 15000)

model_path=$1

for item in "${my_array[@]}"; do
    python render.py -m $model_path -w --skip_test --iteration "$item"
done
