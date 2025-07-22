from market.agent import TradingAgent
import numpy as np
from statistics import mean
from util import Box, Discrete, clamp, ft

# Torch and RL-specific imports.
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

# Internal actor features are (shares held, shares in open orders).
# Environmental market features are configurable.
actor_feat = 2

# Layer initialization for networks.
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """ Defines the actor and critic networks for a PPO agent. """
    def __init__(self, obs_space, act_space, args):
        super().__init__()
        actor_obs_shape, env_obs_shape = (actor_feat,), (obs_space.shape[0],obs_space.shape[1]-actor_feat)
        act_shape = np.prod(act_space.n)

        fc_in = np.prod(actor_obs_shape) + (args.embed if args.encoder == "lstm" else args.levels*2)

        # Features up to actor_feat are internal state repeated along seq dim.
        # Remaining features are environmental, sequenced, and meant for LSTM embedding.
        self.a1 = layer_init(nn.Linear(fc_in, args.netsize))
        self.a2 = layer_init(nn.Linear(args.netsize, args.netsize))
        self.a3 = layer_init(nn.Linear(args.netsize, act_shape), std=0.01)

        self.c1 = layer_init(nn.Linear(fc_in, args.netsize))
        self.c2 = layer_init(nn.Linear(args.netsize, args.netsize))
        self.c3 = layer_init(nn.Linear(args.netsize, 1), std=1.0)

        # Expected input (batch x seq_len x n_features).
        # Expected output (batch x seq_len).
        if args.encoder == "lstm":
            self.lstm = nn.LSTM(input_size = np.array(env_obs_shape[1:]).prod(),
                                hidden_size = args.embed, batch_first = True)

    def actor_fwd(self, x):
        return self.run_fwd(x, self.a1, self.a2, self.a3)

    def critic_fwd(self, x):
        return self.run_fwd(x, self.c1, self.c2, self.c3)

    def run_fwd(self, x, layer1, layer2, layer3):
        if hasattr(self, 'lstm'):
            act_obs, env_obs = x[:,0,:actor_feat], x[:,:,actor_feat:]
            x,_ = self.lstm(env_obs)

            # Take only last sequence output as input to FC.  Repeat internal state.
            x = F.tanh(layer1(torch.cat((act_obs, x[:,-1,:]), dim=1)))
        else:
            x = F.tanh(layer1(x[:,-1,:]))

        return layer3(F.tanh(layer2(x)))

    def get_value(self, x):
        return self.critic_fwd(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor_fwd(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic_fwd(x)


class PPOAgent(TradingAgent):
    def __init__(self, args, obs_low, obs_high, num_act, tag=''):
        """ RL trading agent with continuous state and discrete action spaces.

            State space is gymnasium Box-like created from the extents passed to init.
            In state, features up to actor_feat are internal (repeated along seq dim),
            remainder are environmental.

            Action space is gymnasium Discrete-like created from num_act.

            PPO implementation based on CleanRL (https://github.com/vwxyzjn/cleanrl).
        """

        super().__init__(args.symbol, args.latency, args.interval, lot=args.shares, tag=tag, offset=1e9)
        self.args = args

        # Store parameters that are not part of args or are required by simulation on agent.
        self.levels = args.levels
        self.obs_space, self.act_space = Box(obs_low, obs_high), Discrete(num_act)
        self.anneal_lr, self.norm_adv, self.clip_vloss = True, True, True
        self.clip_coef, self.ent_coef, self.vf_coef = 0.2, 0.01, 0.5
        self.max_grad_norm, self.target_kl = 0.5, None
        self.batch_size = args.num_steps
        self.minibatch_size = int(self.batch_size // args.num_minibatches)


        # Agent network (contains actor and critic).
        self.agent = Agent(self.obs_space, self.act_space, args)
        # list() for parameters?  eps?
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.lr, eps=1e-5)

        # Storage setup (online, so no replay buffer)
        self.obs = torch.zeros((args.num_steps,) + self.obs_space.shape)
        self.actions = torch.zeros((args.num_steps,) + self.act_space.shape)
        self.logprobs = torch.zeros((args.num_steps,))
        self.rewards = torch.zeros((args.num_steps,))
        self.values = torch.zeros((args.num_steps,))

        # Initialize attributes.
        self.eval, self.action_cost = False, 0.0
        self.step, self.episode, self.episode_step = -1, -1, -1
        self.global_step, self.last_global_step, self.print_every_n = -1, -1, 40


    def message (self, ct, msg):
        self.ct = ct
        old_pv = self.portval
        self.handle(ct, msg)

        if msg['type'] == 'lob':
            # Kernel provides env_obs sequence of historical LOB volumes per level.
            env_obs = np.clip(np.array(msg['hist'])/1e4, 0, 1)

            # This agent will not act until there is enough historical information
            # to feed to its recurrent encoder (if present).
            if hasattr(self.agent, 'lstm') and env_obs.shape[0] < self.args.seqlen: return

            # Don't act unless the order book is full.
            min_avail_lev = min(len(self.snap['bid']), len(self.snap['ask']))
            if min_avail_lev < self.levels: return

            # We will only reach here when the agent will take some action.
            self.step += 1
            self.episode_step += 1
            self.global_step += 1

            # Stabilize percent changes by a constant offset for expected VaR.
            offset = 1e6

            # Compute reward of previous action.
            if self.step <= 0: r = None
            else:
                # Immediate P/L as portfolio percent change * 100.  (HFT changes are small.)
                # Action cost is a training penalty, "real" transaction costs handled elsewhere.
                orig_r = 100 * (((self.portval + offset) / (old_pv + offset)) - 1)
                r = orig_r - self.action_cost

                self.total_reward += r

                # Add to previous buffer location.
                self.rewards[self.step-1] = r

            # Compute the new observation.  Snapshot must have the expected number of levels
            # for the state space (controlled via exchange lob subscription).
            next_obs = np.empty(self.obs_space.shape)
            next_obs[:,0] = self.held / self.args.shares
            next_obs[:,1] = self.open / self.args.shares
            next_obs[:,2:] = env_obs
            next_obs = torch.Tensor(next_obs).unsqueeze(0)

            # If the buffer is full (we just placed the last reward, and there is no place
            # for the "current" observation, etc) it is time to do some training.
            if self.step >= self.args.num_steps:

                # bootstrap value if not done
                with torch.no_grad():
                    next_value = self.agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(self.rewards)
                    lastgaelam = 0
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1: nextvalues = next_value
                        else: nextvalues = self.values[t + 1]
                        delta = self.rewards[t] + self.args.gamma * nextvalues - self.values[t]
                        advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * lastgaelam
                    returns = advantages + self.values
        
                # flatten the batch
                b_obs = self.obs.reshape((-1,) + self.obs_space.shape)
                b_logprobs = self.logprobs.reshape(-1)
                b_actions = self.actions.reshape((-1,) + self.act_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = self.values.reshape(-1)
        
                # Optimizing the policy and value network
                b_inds = np.arange(self.batch_size)
                clipfracs = []
                for epoch in range(self.args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, self.batch_size, self.minibatch_size):
                        end = start + self.minibatch_size
                        mb_inds = b_inds[start:end]
        
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()
        
                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
        
                        mb_advantages = b_advantages[mb_inds]
                        if self.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        
                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
                        # Value loss
                        newvalue = newvalue.view(-1)
                        if self.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -self.clip_coef,
                                self.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
        
                        entropy_loss = entropy.mean()
                        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
        
                        self.optimizer.zero_grad()
                        # No backward pass in eval mode.
                        if not self.eval:
                            loss.backward()
                            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                            self.optimizer.step()

                        # Record losses at every update.
                        self.act_losses.append(pg_loss.item())
                        self.crit_losses.append(v_loss.item())
                        self.losses.append(loss.item())
        
                    if self.target_kl is not None and approx_kl > self.target_kl:
                        break

                self.step = 0

                # Sometimes also print an update to the console.
                if self.last_global_step // self.print_every_n < self.global_step // self.print_every_n:
                    print(f"step: {self.global_step}, ct: {ft(ct)}, crit_loss: {self.crit_losses[-1]:.4f}, "
                          f"act_loss: {self.act_losses[-1]:.4f}, loss: {self.losses[-1]:.4f}, "
                          f"ttl_rwd: {self.total_reward:.4f}")
                self.last_global_step = self.global_step

            # Get action and record current information to buffer.
            # If we just trained, we'll have started back at step 0 for next_obs.
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.values[self.step] = value.flatten()
            self.obs[self.step] = next_obs
            self.actions[self.step] = action
            self.logprobs[self.step] = logprob


            # Yield the results, including transaction costs, of the action.
            curr = self.held + self.open              # Shares owned + requested.
            pr_a = action
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



    def reset (self):
        """ Reset any attributes that should not carry across episodes/days. """

        # Increment episode and reset losses and total rewards for reporting purposes.
        self.total_reward = 0
        self.episode += 1
        self.step = 0
        self.act_losses, self.crit_losses, self.losses = [], [], []

    def report_loss (self):
        """ Returns the episode step, global step, actor loss, and critic loss for logging. """
        act_loss = np.nan if len(self.act_losses) == 0 else self.act_losses[-1]
        crit_loss = np.nan if len(self.crit_losses) == 0 else self.crit_losses[-1]
        return self.episode, self.episode_step, self.global_step, act_loss, crit_loss


