#!/bin/bash

my_array=(-8 -9 -10)
#my_array=(-18 -19 -20)
#my_array=(-8 -9 -10 -18 -19 -20 15000)
#my_array=(-100 -101 -102 -103 -104 -105 -106 -107 -108 -109)
#my_array=(-2 -1000 -3000 -5000 -7000 -9000)

model_path=$1

for item in "${my_array[@]}"; do
    python render.py -m $model_path -w --skip_test --iteration "$item"
done
