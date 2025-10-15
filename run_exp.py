""" This is the program to call from the command line, or a slurm batch script,
    to initiate an experiment of potentially many simulations.  It expects at
    least one parameter: --exp <experiment>, which must be the base filename
    of a script in the ./experiments directory.

    Arguments collected in this program are universal to all experiments.
    Experiments may also allow/require additional arguments in ./experiments. """

import argparse, importlib, os, time
from random import randint
from util import set_manual_seed

parser = argparse.ArgumentParser(description='Runs simulation experiments.')

# Experiment selection arguments.
exp_group = parser.add_argument_group('Experiment selection arguments.')
parser.add_argument('--exp', required=True,
                    help='Name of file in experiments directory.')
parser.add_argument('--exp_help', action='store_true',
                  help='Print help for selected experiment (instead of this help).')

# Basic arguments for all experiments.
basic_group = parser.add_argument_group('Basic arguments')
basic_group.add_argument('--tag', default='experiment', metavar='USER_TAG',
                         help='User-supplied tag to identify this experiment or batch.')
basic_group.add_argument('--runtag', default='experiment', metavar='RUN_TAG',
                         help='Optional tag for individual run within batch.')
basic_group.add_argument('--ts', default=None, metavar='TIMESTAMP', help="Optional timestamp for batch.")
basic_group.add_argument('--seed', type=int, default=-1, help="Optional manual seed for PRNG.")
basic_group.add_argument('--datadir', default="./data/lobster", help="Historical data directory.")
basic_group.add_argument('--trips', type=int, default=1, metavar='INT',
                         help='Number of times to train learning agents on each date before'
                              'validating/testing.')
basic_group.add_argument('--fixed', action='store_true',
                         help='Retain first episode random start time offset across episodes.')
basic_group.add_argument('--replay', default='yes', metavar='YES_NO',
                         help='After agent start time, should historical orders be replayed to the exchange?')
basic_group.add_argument('--fund', default='history', metavar='FUND_TYPE',
                         help='Fundamental price-time series: fixed, history.')

# Exchange-related arguments for all experiments.
exch_group = parser.add_argument_group('Exchange-related arguments')
exch_group.add_argument('--symbol', default='AAPL', metavar='SYMBOL',
                         help='Symbol to train.')
exch_group.add_argument('--seqlen', type=int, default=20, metavar='INT',
                        help='Length of historical sequence to offer agents in LOB messages.')
exch_group.add_argument('--lobintvl', type=float, default=5e9, metavar='FLOAT',
                        help='Interval between LOB snapshots for history (ns).')
exch_group.add_argument('--levels', type=int, default=10, metavar='INT',
                        help='Levels of bid/ask volume in LOB snapshot history.')

# Date-related arguments for all experiments.
date_group = parser.add_argument_group('Date-related arguments.')
date_group.add_argument('--train_dates', nargs='+', default=['2019-12-30'], metavar='YYYY-MM-DD',
                        help='YYYY-MM-DD dates on which to train (round robin through multiple dates)')
date_group.add_argument('--val_dates', nargs='+', default=['2019-12-30'], metavar='YYYY-MM-DD',
                        help='YYYY-MM-DD dates on which to validate (model selection after training)')
date_group.add_argument('--test_dates', nargs='+', default=['2019-12-30'], metavar='YYYY-MM-DD',
                        help='YYYY-MM-DD dates on which to test (OOS test after model selection)')

# Time-related arguments for all experiments.
time_group = parser.add_argument_group('Time-related arguments.')
time_group.add_argument('--sim_start', type=float, default=0.50, metavar='FLOAT',
                        help='Simulation start time for pre-market orders (hours since midnight).')
time_group.add_argument('--exch_open', type=float, default=9.50, metavar='FLOAT',
                        help='Exchange open time (hours since midnight).')
time_group.add_argument('--exch_close', type=float, default=16.00, metavar='FLOAT',
                        help='Exchange close time (hours since midnight).')
time_group.add_argument('--train_start', type=float, default=10.00, metavar='FLOAT',
                        help='Start time for experimental agent(s) during training.')
time_group.add_argument('--train_end', type=float, default=11.00, metavar='FLOAT',
                        help='End time for the simulation on each date during training.')
time_group.add_argument('--val_start', type=float, default=11.00, metavar='FLOAT',
                        help='Start time for experimental agent(s) during validation.')
time_group.add_argument('--val_end', type=float, default=11.50, metavar='FLOAT',
                        help='End time for the simulation on each date during validation.')
time_group.add_argument('--test_start', type=float, default=11.50, metavar='FLOAT',
                        help='Start time for experimental agent(s) during testing.')
time_group.add_argument('--test_end', type=float, default=12.00, metavar='FLOAT',
                        help='End time for the simulation on each date during testing.')


args, exp_args = parser.parse_known_args()

# Create a results subdirectory for the experiment with the tag and a unique timestamp.
# For batches, timestamp can be passed in to ensure all files flow to one place.
run_ts = args.ts if args.ts is not None else str(int(time.time()))

print(f"Experiment: {args.exp}, tag: {args.tag}, timestamp: {run_ts}")

args.result_dir = f"results/{args.tag}_{run_ts}"
os.makedirs(args.result_dir, exist_ok=True)

# Transform time arguments from fractional hours (easier to specify on command line)
# to nanoseconds since midnight (what the simulation requires).
args.sim_start *= 3.6e12; args.exch_open *= 3.6e12; args.exch_close *= 3.6e12
args.train_start *= 3.6e12; args.train_end *= 3.6e12; args.val_start *= 3.6e12
args.val_end *= 3.6e12; args.test_start *= 3.6e12; args.test_end *= 3.6e12

# Transform other arguments to a standardized form.
args.symbol = args.symbol.upper()

# Initialize the requested seed, or select and remember a random one.
# If no seed requested, select and remember one.
if args.seed < 0: args.seed = randint(0, 2**32 - 1)
set_manual_seed(args.seed)

# Launch the requested experiment.
experiment = importlib.import_module(f'experiments.{args.exp}', package=None)
experiment.experiment(args, exp_args)

