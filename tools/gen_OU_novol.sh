#!/bin/bash

# Bash script to generate 10 LOBSTER-style CSV files trending up from 100 to 101

python gen_OU_data.py \
	--mu 100 \
	--Q0 100 \
	--sigma 0 \
	--gamma 1 \
	--shocks_per_day 0 \
	--trend 1 \
	--output_file "data/OUUPnovol_2019-12-30__message_0.csv" \
	--seed 1

python gen_OU_data.py \
	--mu 100 \
	--Q0 100 \
	--sigma 0 \
	--gamma 1 \
	--shocks_per_day 0 \
	--trend 0 \
	--output_file "data/OULVLnovol_2019-12-30__message_0.csv" \
	--seed 1

python gen_OU_data.py \
	--mu 100 \
	--Q0 100 \
	--sigma 0 \
	--gamma 1 \
	--shocks_per_day 0 \
	--trend -1 \
	--output_file "data/OUDNnoovol_2019-12-30__message_0.csv" \
	--seed 1
