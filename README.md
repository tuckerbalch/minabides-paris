# minabides
minabides is a simpler, more focused financial market simulation, inspired by the ABIDES simulation, but built from scratch.
It is focused on multi-agent interaction at a simulated exchange which is also receiving historical replay of a complete
Nasdaq order stream, and particularly useful for deep reinforcement learning agents.

minabides is intended as an easy to understand jumping off point for those who want to experiment with market simulation
or to do their own research.  It should be simple enough for classroom and assignment use in a computational finance or
AI/ML course.

## Partial feature list
Particularly when contrasted to ABIDES, minabides has several distinct features:
* Full Nasdaq order replay to the simulated exchange
* Cache and fast-forward history to a designated start time, dramatically speeding up repeat experiments
* Simulation automatically delays all agents according to their actual computational "thinking" time
* Agents follow a (relatively) realistic model for latency and jitter in all their communications
* Working continuous RL agent included, based on cleanrl
* Very easy to create new experimental configurations (experiments/background.py is only about ten lines of code ignoring imports and comments)
* Simple descriptive statistics and plots are built in, pulling from the built in logging functionality
* Support for large parallel experiments on a slurm high performance computing cluster
* For learning experiments, training, validation, and out of sample testing are conducted automatically


## Getting Started
Create a virtual environment and install the required packages.
```bash
python3.12 -mvenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running an experiment
Open a command shell to the base minabides directory, activate your virtual environment, then:
```bash
python run_exp --exp background
```
This runs a single simulation using all available background agents.  Alternatively:
```bash
python run_exp --exp ddpg_td3
```
This runs a single simulation using a deep reinforcement learning agent with continuous environment and actions.

## What are all the moving parts?
These are important to consider when first launching an experiment:
* **run_exp.py** The main program to launch any minabides simulation.
* **experiments/** The folder that holds experiment files passed to the --exp parameter of run_exp.py.
* **simulation.py** The core simulation logic initiated by an experiment after agent configuration.

Additional base simulation components:
* **history.py** Contains the logic to replay, cache, and fast-forward historical order stream data.
* **market.py** Holds all of the background market agents: exchange, market maker, momentum, noise, order book imbalance, and value, and their base classes.
* **rl.py** Holds the reinforcement learning trading agents and related code: DDPG and TD3.
* **util.py** Contains utility methods used by other logic.

After running a simulation, you may want:
* **collect_logs.py** Pulls together the logs from a whole batch of experiments to a single CSV for further analysis.
* **stats.py** Reads the above CSV and produces tabular descriptive statistics and simple plots.

The other folders are:
* **csvs/** Where collect_logs.py puts its CSVs.
* **data/** Where history.py expects historical data to be.
* **images/** Where stats.py writes any plot images.
* **log/** Where the main simulation logs end-of-period performance for each agent.
* **output/** Where the main simulation records all printed output (a tee of stdout).
* **results/** Holds output and error files when running batch experiments under slurm cluster management.

If you have access to a slurm cluster, you may also like:
* **hp_search.sh** A slurm file for use with sbatch to launch hundreds of hyperparameter search experiments in parallel.

## Getting data
minabides does not currently come with data due to licensing constraints.  It has the ability, out of the box
to work with level zero message files from [LOBSTER] (https://lobsterdata.com), which is processed from Nasdaq
TotalView data and licensed for academic use.  Simply place message files in the ./data/lobster folder.

Note that premarket data is expected starting from midnight, so the history agent can ensure the correct opening
limit order book is reconstructed in the exchange.
