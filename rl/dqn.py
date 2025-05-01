from market.agent import TradingAgent
import numpy as np
from statistics import mean
from util import Box, Discrete, ReplayBuffer, clamp, ft

# Torch and RL-specific imports.
import torch
from torch import nn, optim
from torch.nn import functional as F

# Internal actor features are (shares held, shares in open orders).
# Environmental market features are configurable.
actor_feat = 2


class QNetwork(nn.Module):
    """ Defines the network that estimates the Q value of each action, given a state. """
    def __init__(self, obs_space, act_space, args):
        super().__init__()
        actor_obs_shape, env_obs_shape = (actor_feat,), (obs_space.shape[0],obs_space.shape[1]-actor_feat)
        act_shape = np.prod(act_space.n)

        fc_in = np.prod(actor_obs_shape) + (args.embed if args.encoder == "lstm" else args.levels*2)

        # Features up to actor_feat are internal state repeated along seq dim.
        # Remaining features are environmental, sequenced, and meant for LSTM embedding.
        self.fc1 = nn.Linear(fc_in, args.netsize)
        self.fc2 = nn.Linear(args.netsize, args.netsize)
        self.fc3 = nn.Linear(args.netsize, act_shape)

        # Expected input (batch x seq_len x n_features).
        # Expected output (batch x seq_len).
        if args.encoder == "lstm":
            self.lstm = nn.LSTM(input_size = np.array(env_obs_shape[1:]).prod(),
                                hidden_size = args.embed, batch_first = True)

    def forward(self, x):
        if hasattr(self, 'lstm'):
            act_obs, env_obs = x[:,0,:actor_feat], x[:,:,actor_feat:]
            x,_ = self.lstm(env_obs)

            # Take only last sequence output as input to FC.  Repeat actor internal state.
            x = F.relu(self.fc1(torch.cat((act_obs, x[:,-1,:]), dim=1)))
        else:
            x = F.relu(self.fc1(x[:,-1,:]))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQNAgent(TradingAgent):
    def __init__(self, args, obs_low, obs_high, num_act, tag=''):
        """ RL trading agent with continuous state and discrete action spaces.

            State space is gymnasium Box-like created from the extents passed to init.
            In state, features up to actor_feat are internal (repeated along seq dim),
            remainder are environmental.

            Action space is gymnasium Discrete-like created from num_act.

            DQN implementation based on CleanRL (https://github.com/vwxyzjn/cleanrl).
        """

        super().__init__(args.symbol, args.latency, args.interval, lot=args.shares, tag=tag, offset=1e9)
        self.args = args

        # Store parameters that are not part of args or are required by simulation on agent.
        self.levels = args.levels
        self.obs_space, self.act_space = Box(obs_low, obs_high), Discrete(num_act)
        self.rb = ReplayBuffer(maxlen = args.rbuf, obs_shape = self.obs_space.shape,
                               act_shape = self.act_space.shape, discrete_actions = True)
    
        # Create the Q and target networks.
        self.q_network = QNetwork(self.obs_space, self.act_space, args)
        self.target_network = QNetwork(self.obs_space, self.act_space, args)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer to train Q network.  (Target network is not trained.)
        self.optimizer = optim.Adam(list(self.q_network.parameters()), lr=args.lr)
    
        # Initialize attributes.
        self.eval, self.obs, self.act, self.action_cost = False, None, None, 0.0
        self.episode, self.episode_step, self.global_step = -1, -1, -1
        self.learn_start_step = args.startstep


    def message (self, ct, msg):
        self.ct = ct
        old_pv = self.portval
        self.handle(ct, msg)

        if msg['type'] == 'lob':
            # Kernel provides env_obs sequence of historical LOB volumes per level.
            env_obs = np.clip(np.array(msg['hist'])/1e4, 0, 1)

            # This agent will not act until there is enough historical information
            # to feed to its recurrent encoder.
            if env_obs.shape[0] < self.args.seqlen: return

            self.episode_step += 1
            self.global_step += 1

            # Stabilize percent changes by a constant offset for expected VaR.
            offset = 1e6

            # Compute reward of previous action.
            if old_pv is None or self.obs is None or self.act is None: r = None
            else:
                # Immediate P/L as portfolio percent change * 100.  (HFT changes are small.)
                # Action cost is a training penalty, "real" transaction costs handled elsewhere.
                orig_r = 100 * (((self.portval + offset) / (old_pv + offset)) - 1)
                #self._debug(f"a[0] = {self.act[0].item():.4f}, r = {r:.4f} at {ft(ct)}.")
                r = orig_r - self.action_cost
                if self.args.rldebug: print(f"h = {self.obs[0,0]:.4}, a = {self.act}, "
                                            f"orig r = {orig_r:.4f}, r = {r:.4f} at {ft(ct)}.")

                self.total_reward += r

            # Compute the new observation.  Snapshot must have the expected number of levels
            # for the state space (controlled via exchange lob subscription).
            next_obs = np.empty(self.obs_space.shape)
            next_obs[:,0] = self.held / self.args.shares
            next_obs[:,1] = self.open / self.args.shares
            next_obs[:,2:] = env_obs

            # Record to replay buffer using previous and current information.
            real_next_obs = next_obs.copy()
            if r is not None and self.act is not None:
                self.rb.add(self.obs, self.act, real_next_obs, r)

            # As soon as it is saved, we can update the observation.
            # Used as "current" to make decisions now, and saved as "previous" for next time.
            self.obs = next_obs

            # Don't act unless the order book is full.
            min_avail_lev = min(len(self.snap['bid']), len(self.snap['ask']))
            if min_avail_lev < self.levels: return

            # Query the learner for an action.  (Always random at first.)
            epsilon = linear_schedule(self.args.start_e, self.args.end_e,
                                      self.args.explore_frac * self.args.total_timesteps, self.global_step)
            if np.random.uniform() < epsilon:
                action = self.act_space.sample()
            else:
                q_values = self.q_network(torch.Tensor(self.obs).unsqueeze(0))
                action = torch.argmax(q_values, dim=1).item()

            # Although we may modify the results of the action (e.g. portfolio holding limit),
            # we always record the requested action for replay purposes.
            self.act = action

            # Yield the results, including transaction costs, of the action.
            curr = self.held + self.open              # Shares owned + requested.
            if action == 0:
                a = -self.args.trade
                self.action_cost = 0.01               # approx 1 bp (r is * 100)
            elif action == 1:
                a = 0
                self.action_cost = 0
            elif action == 2:
                a = self.args.trade
                self.action_cost = 0.01               # approx 1 bp (r is * 100)
            prop = curr + a                           # Proposed shares after action.

            # Adjust action so proposed shares fall within holding limits.
            if prop > self.args.shares: a = self.args.shares - curr
            elif prop < -self.args.shares: a = -self.args.shares - curr

            # Record transaction cost, if any.
            trans_cost = self.args.trans_cost * abs(a) * (self.ask if a >= 0 else self.bid)
            self.cost(trans_cost)
            yield self.place(a)


            # Do some training.
            if self.global_step == self.learn_start_step:
                print("Starting to train batches every", self.args.train_freq, "steps as of step", self.global_step)

            if self.global_step >= self.learn_start_step and self.global_step % self.args.train_freq == 0:
                data = self.rb.sample(self.args.batchsize)

                with torch.no_grad():
                    target_max, _ = self.target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + self.args.gamma * target_max

                old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                self.optimizer.zero_grad()
                if not self.eval:   # Eval mode means no backprop.
                    loss.backward()
                    self.optimizer.step()

                # Only update target periodically.
                if self.global_step % self.args.target_freq == 0:
                    for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

                # Record losses at every q_network update.
                self.losses.append(loss.item())

                # Print an update to the console each time we train a batch.
                print(f"ct: {ft(ct)}, loss: {loss.item():.4f}, ttl_rwd: {self.total_reward:.4f}")


    def reset (self):
        """ Reset any attributes that should not carry across episodes/days. """
        # Increment episode and reset losses and total rewards for reporting purposes.
        self.total_reward = 0
        self.episode += 1
        self.losses = []

    def report_loss (self):
        """ Returns the episode step, global step, actor loss, and critic loss for logging. """
        # Note: this RL agent has no critic.
        loss = np.nan if len(self.losses) == 0 else self.losses[-1]
        return self.episode, self.episode_step, self.global_step, loss, np.nan

