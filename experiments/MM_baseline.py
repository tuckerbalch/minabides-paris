
""" 
    A baseline configuration of agents for experiments with Market Makers.

    Original comment: This is an example experiment in which only non-learning
    "background" agents trade with an LOB populated by real
    historical orders, plus orders from the agents themselves.

    The experiment should not be directly run.  Rather, it should be
    passed as the --exp parameter to run_exp.py in the main project
    directory.

    Also note that only experiment-specific arguments are parsed in
    this file.  There are many basic arguments required for all experiments,
    which are handled in run_exp.py before handing control to this file.
    It is not necessary to add every agent parameter to argparse, only those
    that may be part of a hyperparameter search.

    It is strongly suggested to copy a working experiment as the model
    for any new experiments.
"""

if __name__ == "__main__":
    print("This file cannot be run directly.  Please pass it to the --exp argument "
          "of run_exp.py in the main project directory.")
    exit()

def experiment(base_args, exp_args):

    # Built-in or PyPy imports.
    import argparse

    # Custom imports from this project.
    from market.background import MarketMakerAgent_AS, MarketMakerAgent, MomentumAgent, NoiseAgent, \
                                  OrderBookImbalanceAgent, ValueAgent
    from market.exchange import ExchangeAgent
    from simulation import run_experiment

    # This experiment requires no additional command-line parameters.
    args = base_args

    # Configure the base agents for the experiment.  Exchange must be agent zero.
    # "Base" agents are those always present for this experiment.
    #
    # These are the baseline configuration of agents that work well with
    # our MM experiments.
    base_ag = [ ExchangeAgent([args.symbol], args) ] + \
        [ MarketMakerAgent_AS(args.symbol, 1e1, 2e8, 100) ] + \
        [ MomentumAgent(args.symbol, 5e6, 1e9, 100) for i in range(5)] + \
        [ NoiseAgent(args.symbol, 12e6, 10e9) for i in range(10)] + \
        [ ValueAgent(args.symbol, 3e6, 5e8, 100) for i in range(5)]
    
    # old, not used for now
    # [ OrderBookImbalanceAgent(args.symbol, 1e3, 2e8, 100, 50) for i in range(5)] + \

  #[ MarketMakerAgent(args.symbol, 1e1, 2e8, 100, 10) ] + \
  
    # This experiment uses no learning agents.
    learn_ag = []

    # Launch the experiment.
    run_experiment(base_ag, learn_ag, args)

