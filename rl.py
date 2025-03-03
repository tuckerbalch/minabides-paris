from collections import namedtuple
from market import TradingAgent
import numpy as np
from statistics import mean
from util import clamp, ft

# Torch and RL-specific imports.
import torch
from torch import nn, optim
from torch.nn import functional as F

# Internal actor features are (shares held, shares in open orders).
# Environmental market features are configurable.
actor_feat = 2

# Simple named tuple type for replay samples.
ReplaySample = namedtuple('ReplaySample', ['observations', 'actions', 'next_observations', 'rewards'])


class Box():
    """ A very simple continuous space in the style of the gymnasium Box.
        Does not support unbounded spaces. """
    def __init__(self, low, high):
        self.low, self.high = low, high
        self.shape = self.low.shape

    def sample(self):
        """ Uniformly randomly samples from this bounded space. """
        return np.random.uniform(low=self.low, high=self.high, size=self.shape)


class ReplayBuffer():
    """ A very simple replay buffer in the style of stable-baselines3. """
    def __init__(self, maxlen, obs_shape, act_shape):
        self.s = np.empty((maxlen, *obs_shape), dtype=np.float32)
        self.a = np.empty((maxlen, *act_shape), dtype=np.float32)
        self.s_prime = np.empty((maxlen, *obs_shape), dtype=np.float32)
        self.r = np.empty((maxlen), dtype=np.float32)

        self.n = 0
        self.full = False

    def add(self, s, a, s_prime, r):
        self.s[self.n] = np.array(s)
        self.a[self.n] = np.array(a)
        self.s_prime[self.n] = np.array(s_prime)
        self.r[self.n] = np.array(r)

        self.n = self.n + 1 % self.s.shape[0]
        if self.n == 0: self.full = True

    def sample(self, num_samples):
        idx = np.random.randint(0, self.s.shape[0] if self.full else self.n, size=num_samples)
        return ReplaySample(torch.tensor(self.s[idx]), torch.tensor(self.a[idx]),
                            torch.tensor(self.s_prime[idx]), torch.tensor(self.r[idx]))


class ActorCritic(nn.Module):
    """ Defines the actor and critic networks for a DDPG/TD3 agent. """
    def __init__(self, obs_space, act_space, args, scale=1.0, critic=False):
        super().__init__()
        self.critic, self.scale = critic, scale
        actor_obs_shape, env_obs_shape = (actor_feat,), (obs_space.shape[0],obs_space.shape[1]-actor_feat)

        # Features up to actor_feat are internal state repeated along seq dim.
        # Remaining features are environmental, sequenced, and meant for LSTM embedding.

        # Critic also receives an act_shape action as input.  Actor does not.
        # Critic outputs one scalar.  Actor outputs act_shape.
        act_shape = np.prod(act_space.shape)

        self.fc1 = nn.Linear(np.prod(actor_obs_shape) + args.embed + (act_shape if critic else 0),
                             args.netsize)
        self.fc2 = nn.Linear(args.netsize, args.netsize)
        self.fc3 = nn.Linear(args.netsize, 1 if critic else act_shape)

        self.ln1 = nn.LayerNorm(args.netsize)
        self.ln2 = nn.LayerNorm(args.netsize)

        # Expected input (batch x seq_len x n_features).
        # Expected output (batch x seq_len).
        self.lstm_feat_embed = nn.LSTM(input_size = np.array(env_obs_shape[1:]).prod(),
                                hidden_size = args.embed, batch_first = True)

    def forward(self, x, a=None):
        act_obs, env_obs = x[:,0,:actor_feat], x[:,:,actor_feat:]
        x,_ = self.lstm_feat_embed(env_obs)

        # Take only last sequence output as input to FC.  Repeat actor internal state.
        # Critic network requires action input.  Actor does not.
        if self.critic: x = F.relu(self.ln1(self.fc1(torch.cat((act_obs, x[:,-1,:], a), dim=1))))
        else: x = F.relu(self.ln1(self.fc1(torch.cat((act_obs, x[:,-1,:]), dim=1))))

        x = F.relu(self.ln2(self.fc2(x)))

        # For actor only, use tanh final activation, then scale to action space.
        if self.critic: x = self.fc3(x)
        else: x = F.tanh(self.fc3(x)) * self.scale

        return x


class DDPGAgent(TradingAgent):
    def __init__(self, args, obs_low, obs_high, act_low, act_high, tag=''):
        """ RL trading agent with continuous state and action spaces.

            State and action spaces are gymnasium Box-likes created from the
            extents passed to init.  In state, features up to actor_feat are
            internal (repeated along seq dim), remainder are environmental.

            DDPG implementation based on CleanRL (https://github.com/vwxyzjn/cleanrl).
            Also contains option for TD3 based on CleanRL.
        """

        super().__init__(args.symbol, args.latency, args.interval, lot=args.shares, tag=tag, offset=1e9)
        self.args = args

        # Store parameters that are not part of args or are required by simulation on agent.
        self.levels = args.levels
        self.obs_space, self.act_space = Box(obs_low, obs_high), Box(act_low, act_high)
        self.td3 = True if args.rlagent.upper() == "TD3" else False
        self.action_scale = torch.tensor((self.act_space.high - self.act_space.low) / 2.0, dtype=torch.float32)
        self.rb = ReplayBuffer(maxlen = args.rbuf, obs_shape = self.obs_space.shape, act_shape = self.act_space.shape)
    
        # Actor network.
        self.actor = ActorCritic(self.obs_space, self.act_space, args, scale=self.action_scale)

        # Critic network(s).  Second critic for TD3 only.
        self.qf1 = ActorCritic(self.obs_space, self.act_space, args, critic=True)
        self.qf2 = ActorCritic(self.obs_space, self.act_space, args, critic=True)

        # Actor network target.
        self.target_actor = ActorCritic(self.obs_space, self.act_space, args, scale=self.action_scale)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Critic network target(s).  Second critic target for TD3 only.
        self.qf1_target = ActorCritic(self.obs_space, self.act_space, args, critic=True)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target = ActorCritic(self.obs_space, self.act_space, args, critic=True)
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Optimizers to train actor and critic.  (Targets are not trained.)
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=args.lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.lr)
    
        # Initialize attributes.
        self.eval, self.obs, self.act = False, None, None
        self.episode, self.episode_step, self.global_step = -1, -1, -1
        self.print_every_n_policy_updates = int(20/args.polfreq)
        self.learn_start_step = args.startstep
        self.noise_clip = 0.5


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
                r = 100 * (((self.portval + offset) / (old_pv + offset)) - 1)
                self._debug(f"a[0] = {self.act[0].item():.4f}, r = {r:.4f} at {ft(ct)}.")

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
            if self.global_step < self.learn_start_step:
                actions = self.act_space.sample()
            else:
                with torch.no_grad():
                    # Add batch dimension to single observation.
                    actions = self.actor(torch.Tensor(self.obs).unsqueeze(0)).squeeze(0)
                    actions += torch.normal(0, self.args.expnoise, size=(1,))
                    actions = actions.cpu().numpy().clip(self.act_space.low, self.act_space.high)

            # Although we may modify the results of the action (e.g. portfolio holding limit),
            # we always record the requested action for replay purposes.
            self.act = actions

            # Yield the results of the action.
            curr = self.held + self.open              # Shares owned + requested.
            a = int(self.args.shares * actions[0])         # The trade quantity action.
            prop = curr + a                           # Proposed shares after action.

            # Adjust action so proposed shares fall within holding limits.
            if prop > self.args.shares: a = self.args.shares - curr
            elif prop < -self.args.shares: a = -self.args.shares - curr

            yield self.place(a)


            # Do some training.
            if self.global_step == self.learn_start_step:
                print("Starting to train a batch at every step as of step", self.global_step)

            if self.global_step >= self.learn_start_step:
                data = self.rb.sample(self.args.batchsize)

                with torch.no_grad():
                    # Always update critic networks.
                    if self.td3:
                        clipped_noise = (torch.randn_like(data.actions) * self.args.polnoise).clamp(
                                        -self.noise_clip, self.noise_clip) * self.action_scale

                        next_state_actions = (self.target_actor(data.next_observations) + clipped_noise).clamp(
                                self.act_space.low[0], self.act_space.high[0]
                        )

                        # Use the smaller Q estimate between the two critic target networks.
                        qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                        qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                        next_q_value = data.rewards.flatten() * self.args.gamma * (min_qf_next_target).view(-1)

                        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)

                    else:
                        next_state_actions = self.target_actor(data.next_observations)
                        qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                        next_q_value = data.rewards.flatten() * self.args.gamma * (qf1_next_target).view(-1)

                # Use of first critic does not vary between DDPG and TD3.
                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                if self.td3: qf_loss = qf1_loss + qf2_loss
                else: qf_loss = qf1_loss

                self.q_optimizer.zero_grad()
                if not self.eval:   # Eval mode means no backprop.
                    qf_loss.backward()
                    self.q_optimizer.step()

                # Only update actor and targets periodically.
                if self.global_step % self.args.polfreq == 0:
                    actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
                    self.actor_optimizer.zero_grad()
                    if not self.eval:
                        actor_loss.backward()
                        self.actor_optimizer.step()

                    # Update target networks.
                    for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                    if self.td3:    # Second critic for TD3 only.
                        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

                    # Record losses at every policy (actor) update.
                    self.act_losses.append(actor_loss.item())
                    self.qf_losses.append(qf_loss.item())

                    # Sometimes also print an update to the console.
                    if self.global_step % (self.args.polfreq * self.print_every_n_policy_updates) == 0:
                        print(f"ct: {ft(ct)}, qf_loss: {qf_loss.item():.4f}, act_loss: "
                              f"{actor_loss.item():.4f}, ttl_rwd: {self.total_reward:.4f}")


    def new_day (self, start, evaluate=False):
        """ Reset any attributes that should not carry across episodes/days. """
        self.random_start(start)

        # Increment episode and reset losses and total rewards for reporting purposes.
        self.total_reward = 0
        self.episode += 1
        self.qf_losses, self.act_losses = [], []

        # Update attributes passed as parameters.
        self.eval = evaluate


