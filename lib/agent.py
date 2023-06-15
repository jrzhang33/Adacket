
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from lib.memory import SequentialMemory
from lib.utils import to_numpy, to_tensor
criterion = nn.MSELoss()
USE_CUDA = torch.cuda.is_available()
class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.nb_states = 75 #70
        self.fc1 = nn.Linear(self.nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_input):
        if x_input.shape[0] * x_input.shape[1] == self.nb_states:
            x = torch.flatten(x_input)
            x = x.unsqueeze(0)
        else:
            x = x_input
        x = (x - x.mean()) / x.std()
        assert x.std()> 1e-3
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out) 

        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_actions + nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, xs):
        x, a = xs
        x = (x - x.mean()) / x.std()
        out = torch.cat([x, a],-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DDPG(object):
    def __init__(self, nb_states, nb_actions, args,channels):
        self.channels = channels

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            # 'init_w': args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_a)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_c)

        self.hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        self.hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        # self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu,
        #                                                sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
        self.lbound = args.lbound  # args.lbound
        self.rbound = args.rbound  # args.rbound
        self.lcbound = args.lcbound  # args.lbound
        self.rcbound = args.rcbound  # args.rbound
        self.lobound = args.lobound  # args.lbound
        self.robound = args.robound  # args.rbound
        # noise
        self.init_delta = args.init_delta
        self.delta_decay = args.delta_decay
        self.warmup = args.warmup

        #
        self.epsilon = 1.0
        # self.s_t = None  # Most recent state
        # self.a_t = None  # Most recent action
        self.is_training = True

        #
        if USE_CUDA: self.cuda()

        # moving average baseline
        self.moving_average = None
        self.moving_alpha = 0.5  # based on batch, so small

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # normalize the reward
        batch_mean_reward = np.mean(reward_batch)
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        reward_batch -= self.moving_average
        # if reward_batch.std() > 0:
        #     reward_batch /= reward_batch.std()

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                to_tensor(next_state_batch),
                self.actor_target(to_tensor(next_state_batch)),
            ])

        target_q_batch = to_tensor(reward_batch) + \
                         self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t, s_t1, a_t, done):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, done)  # save to memory
            # self.s_t = s_t1

    def random_action(self):
        action1 = np.random.uniform(self.lbound, self.rbound, 1)
        action2 = np.random.uniform(self.lcbound, self.rcbound, 1)
        action3 = np.random.uniform(self.lobound, self.robound, 1)
        action4 = np.random.uniform(self.lobound, self.robound, 1)
        # self.a_t = action
        return np.concatenate([[action1], [action2], [action3], [action4]], axis = 1)

    def select_action(self, s_t, episode):
        actiond = np.array(self.actor(to_tensor(np.array(s_t))).detach().cpu())
        action1 = actiond[:,0]
        action1 = action1 + np.random.normal(-0.1, 0.1, action1.size)
        action1 = np.clip(action1, self.lbound, self.rbound)
        action2 = actiond[:,1]
        action2 = action2 + np.random.normal(-0.1, 0.1, action2.size)
        action2 = np.clip(action2, self.lbound, self.rbound)

        action3 = actiond[:,2]
        action3 = action3 + np.random.normal(-0.05, 0.05, action3.size)
        action3 = np.clip(action3, self.lbound, self.rbound)       
        
        action4 = actiond[:,3]
        action4 = action1 + np.random.normal(-0.1, 0.1, action1.size)
        action4 = np.clip(action1, self.lbound, self.rbound)
        return np.concatenate([[action1], [action2], [action3], [action4]], axis = 1)

    def reset(self, obs):
        pass
        # self.s_t = obs
        # self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def sample_from_truncated_normal_distribution(self, lower, upper, mu, sigma, size=1):
        from scipy import stats
        return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)


