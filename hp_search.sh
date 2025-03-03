#!/bin/bash
#SBATCH --output=results/%x-%j.out  # Output file
#SBATCH --error=results/%x-%j.err   # Error file
#XBATCH --exclusive=user	    # Do not share nodes with other users.

### Adapt to your cluster configuration.
#SBATCH --nodes=6
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5GB

### Can handle any number of permutations and run "tasks" number at a time,
### queuing automatically, so no parameters needed except an identifier.
### Just put everything in this script's for loops.

if [ $# -lt 1 ]; then
  echo $0: 'usage: sbatch <this_script> <run_tag>'
  exit 1
fi

runid="$1"

# Source your virtual environment.
source ./venv/bin/activate

# Initialize task num for separate log file generation.
tasknum=1

# How many background agents of each type should be included?
# (ddpg_td3 example only, background example is hardcoded)
bg_mkt=1      # market maker
bg_mom=5      # momentum
bg_nse=10     # noise
bg_obi=1      # order book imbalance
bg_val=10     # value

### To avoid race conditions, when running a new date or time
### for the first time, consider a single run per combination
### of symbol + date + time, with a single training trip to
### cache all new data files.

# How many times to repeat each configuration in the search.
repeats=({1..1})     # for caching or installation testing
#repeats=({1..192})  # how many times to repeat each experiment

# The hyperparameters over which you want to search.
symbols=("aapl" "intc" "msft")
dates=("2019-12-30")

for repeat in "${repeats[@]}";
do
  for symbol in "${symbols[@]}";
  do
    for date in "${dates[@]}";
    do
      # Example reinforcement learning experiment.
      #srun -u --exclusive=user --mem-per-cpu 5GB -N1 -n1 /usr/bin/time -v python -u run_exp.py --exp ddpg_td3 --train_dates $date --val_dates $date --test_dates $date --symbol $symbol --bg_mkt $bg_mkt --bg_mom $bg_mom --bg_nse $bg_nse --bg_obi $bg_obi --bg_val $bg_val --models 1 --trips 5 --tag "${runid}_${symbol}_${repeat}_${tasknum}" &

      # Example background agent experiment.
      srun -u --exclusive=user --mem-per-cpu 5GB -N1 -n1 /usr/bin/time -v python -u run_exp.py --exp background --train_dates $date --symbol $symbol --trips 5 --tag "${runid}_${symbol}_${repeat}_${tasknum}" &
      tasknum=$((tasknum + 1))
    done
  done
done

wait
      
echo Done.
