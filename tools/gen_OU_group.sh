#!/bin/bash

# Bash script to generate 10 LOBSTER-style CSV files trending up from 100 to 101

for i in {1..10}
do
    python gen_OU_data.py \
        --mu 100 \
        --Q0 100 \
        --trend 1 \
        --output_file "data/OUUP${i}_2019-12-30__message_0.csv" \
        --seed $i
done

for i in {1..10}
do
    python gen_OU_data.py \
        --mu 100 \
        --Q0 100 \
        --trend 0 \
        --output_file "data/OULVL${i}_2019-12-30__message_0.csv" \
        --seed $i
done

for i in {1..10}
do
    python gen_OU_data.py \
        --mu 100 \
        --Q0 100 \
        --trend -1 \
        --output_file "data/OUDN${i}_2019-12-30__message_0.csv" \
        --seed $i
done
