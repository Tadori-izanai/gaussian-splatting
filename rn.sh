#!/bin/bash

#my_array=(-8 -9 -10)
my_array=(-18 -19 -20)
#my_array=(-8 -9 -10 -18 -19 -20 15000)

model_path=$1

for item in "${my_array[@]}"; do
    python render.py -m $model_path -w --skip_test --iteration "$item"
done
