
""" This is an example experiment in which a reinforcement learning
    agent with both continuous states and actions trades against
    an LOB populated by real historical orders.  Depending on the
    configuration, it may be trading interactively against other
    trading agents as well.

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
    import argparse, numpy as np

    # Custom imports from this project.
    import market as m
    from rl import DDPGAgent
    from simulation import run_experiment

    # Read experiment-specific command-line parameters.
    parser = argparse.ArgumentParser(description='This configuration is for a deep RL intraday trading task.',
                                     add_help=False)
    
    # Background agent options for this experiment.
    bg_group = parser.add_argument_group('Background agent options for the RL experiment.')
    bg_group.add_argument('--bg_mkt', type=int, default=0, help="Number of market makers.")
    bg_group.add_argument('--bg_mom', type=int, default=0, help="Number of momentum traders.")
    bg_group.add_argument('--bg_nse', type=int, default=0, help="Number of noise traders.")
    bg_group.add_argument('--bg_obi', type=int, default=0, help="Number of OBI traders.")
    bg_group.add_argument('--bg_val', type=int, default=0, help="Number of value traders.")

    # RL agent arguments specific to this experiment (for hyperparameter search).
    rl_group = parser.add_argument_group('RL agent arguments specific to this experiment.')
    rl_group.add_argument('--rlagent', default="td3", help="RL agent: DDPG or TD3")
    rl_group.add_argument('--expconfig', default="rl", help="Exp. agents: rl, none.")
    rl_group.add_argument('--shares', type=int, default=100, metavar='INT',
                         help='Maximum position size (long or short).')
    rl_group.add_argument('--embed', type=int, default=6, metavar='INT',
                        help='Embed size for environmental observations.')
    rl_group.add_argument('--netsize', type=int, default=64, metavar='INT',
                        help='Network size for agent/policy hidden layers.')
    rl_group.add_argument('--lr', type=float, default=5e-2, metavar='FLOAT',
                        help='Learning rate for agent/policy optimizers.')
    rl_group.add_argument('--models', type=int, default=5, metavar='INT',
                        help='How many models to train and select from in validation.')
    rl_group.add_argument('--interval', type=float, default=30e9, metavar='FLOAT',
                        help='Approx. ns between agent actions.')
    rl_group.add_argument('--rbuf', type=int, default=10000, metavar='INT',
                        help='Size of replay buffer.')
    rl_group.add_argument('--expnoise', type=float, default=0.1, metavar='FLOAT',
                        help='Exploration noise for actor.')
    rl_group.add_argument('--polnoise', type=float, default=0.4, metavar='FLOAT',
                        help='Policy noise for q-network.')
    rl_group.add_argument('--batchsize', type=int, default=32, metavar='INT',
                        help='Batch size for replay training.')
    rl_group.add_argument('--polfreq', type=int, default=2, metavar='INT',
                        help='Interval for policy updates.')
    rl_group.add_argument('--tau', type=float, default=0.02, metavar='FLOAT',
                        help='Interpolation coefficient for target network updates.')
    rl_group.add_argument('--gamma', type=float, default=0.99, metavar='FLOAT',
                        help='Gamma for Bellman equations.')
    rl_group.add_argument('--latency', type=float, default=1e1, metavar='FLOAT',
                        help='Latency for RL agent (ns).')
    rl_group.add_argument('--startstep', type=int, default=5, metavar='FLOAT',
                        help='Random actions before learning starts.')

    # Parse the remaining experimental arguments not handled by run_exp.
    # Combine the basic and experimental arguments to a single namespace.
    exp_args = parser.parse_args(exp_args)
    args = argparse.Namespace(**vars(base_args), **vars(exp_args))

    # Configure the base agents for the experiment.  Exchange must be agent zero.
    # "Base" agents are those always present for this experiment.

    base_ag = [ m.ExchangeAgent([args.symbol], args) ] + \
              [ m.MarketMakerAgent(args.symbol, 1e1, 2e8, 100) for i in range(args.bg_mkt)] + \
              [ m.MomentumAgent(args.symbol, 5e6, 1e9, 100) for i in range(args.bg_mom)] + \
              [ m.NoiseAgent(args.symbol, 12e6, 10e9) for i in range(args.bg_nse)] + \
              [ m.OrderBookImbalanceAgent(args.symbol, 1e3, 2e8, 100, 50) for i in range(args.bg_obi)] + \
              [ m.ValueAgent(args.symbol, 3e6, 5e8, 100) for i in range(args.bg_val)]

    # Configure the learning agents for the experiment.  They are separated out
    # because only one is included in each simulated day.  (args.models models
    # are created as separate agents and round-robined for training, validation,
    # and testing)
    learn_ag = []

    # Configure RL agent if experiment requires it.
    if args.expconfig in ['rl']:

        # Create as many separate models as the experiment requested.
        for _ in range(args.models):
            # Configure observation and action extents for DDPG/TD3 agents.
            obs_low = np.array([[ -1.0, -2.0 ] + [ 0.0 ] * (args.levels * 2)] * args.seqlen)
            obs_high = np.array([[ 1.0, 2.0 ] + [ 1.0 ] * (args.levels * 2)] * args.seqlen)
            act_low = np.array([-2.0])
            act_high = np.array([2.0])

            # Create the agent.
            a = DDPGAgent(args, obs_low, obs_high, act_low, act_high)

            # These models must all share one agent id, since they will be used one at a time,
            # substituting for each other.  All base agents must be added before learning agents.
            a.aid = len(base_ag)
            learn_ag.append(a)

    # Launch the experiment.
    run_experiment(base_ag, learn_ag, args)

