import argparse
import gym
import os
import sys
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
from replay_memory import Memory
from running_state import ZFilter
import math
import numpy as np

from models.policy_network import Policy
from models.value_network import Value

use_gpu = torch.cuda.is_available()

Tensor = torch.cuda.DoubleTensor if use_gpu else torch.DoubleTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
if use_gpu:
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    PI = torch.cuda.DoubleTensor([3.1415926])
else:
    torch.set_default_tensor_type('torch.DoubleTensor')
    PI = torch.DoubleTensor([3.1415926])

env_name = 'Humanoid-v1'
render = False
log_std = 0
gamma = 0.99
tau = 0.95
l2_reg = 1e-3
learning_rate = 0.0003
clip_epsilon = 0.2 # for PPO
seed = 1
min_batch_size = 10000
total_iterations = 1000
log_interval = 1
optim_epochs = 5
optim_batch_size = 64


env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
running_state = ZFilter((state_dim), clip = 5)

policy_net = Policy(state_dim, action_dim)
old_policy_net = Policy(state_dim, action_dim)
value_net = Value(state_dim)

if use_gpu:
    policy_net = policy_net.cuda()
    old_policy_net = old_policy_net.cuda()
    value_net = value_net.cuda()

policy_optim = torch.optim.Adam(policy_net.parameters(), lr = learning_rate)
value_optim = torch.optim.Adam(value_net.parameters(), lr = learning_rate)

def select_action(policy_net, state):
    state = Variable(torch.from_numpy(state).unsqueeze(0).cuda())
    # print(type(state))
    action_mean, _, action_std = policy_net(state)
    action = torch.normal(action_mean, action_std)
    return action

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(PI)) - log_std
    return log_density.sum(1)


def collect_samples(policy_net, min_batch_size):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0

    while (num_steps < min_batch_size):
        state = env.reset()
        state = running_state(state)
        reward_sum = 0
        for t in range(10000):
            action = select_action(policy_net, state)
            action = action.data[0].cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            mask = 0 if done else 1

            memory.push(state, np.array([action]), mask, next_state, reward)

            if render:
                env.render()
            if done:
                break

            state = next_state

        num_steps += (t - 1)
        num_episodes += 1
        reward_batch += reward_sum

    print(num_episodes)
    reward_batch = reward_batch / num_episodes
    
    batch = memory.sample()

    return batch, reward_batch



def gae(states, actions, rewards, values, masks):
    ## initialize the variables
    returns = Tensor(actions.size(0), 1)
    deltas = Tensor(actions.size(0), 1)
    advantages = Tensor(actions.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0

    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        # deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        # prev_value = values.data[i, 0]
        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    advantages = (advantages - advantages.mean())/ advantages.std()
    
    return advantages, targets

def update_parameters(batch, new_lr, new_clip):
    states = Tensor(batch.state)
    actions = torch.squeeze(Tensor(batch.action))
    rewards = Tensor(batch.reward)
    masks = Tensor(batch.mask)
    values = value_net(Variable(states)).data

    policy_optim.lr = new_lr
    value_optim.lr = new_lr

    advantages, targets = gae(states, actions, rewards, values, masks)
    # backup params after computing probs but before updating new params
    #policy_net.backup()
    for old_policy_param, policy_param in zip(old_policy_net.parameters(), policy_net.parameters()):
        old_policy_param.data.copy_(policy_param.data)

    optim_iters = int(math.floor(min_batch_size / optim_batch_size))

    for _ in range(optim_epochs):
        ## shuffle the state action pairs
        perm = LongTensor(np.random.permutation(actions.size(0)))
        states = states[perm]
        actions = actions[perm]
        values = values[perm]
        targets = targets[perm]
        advantages = advantages[perm]

        cur_id = 0
        cur_id_exp = 0

        for _ in range(optim_iters):
            cur_batch_size = min(optim_batch_size, actions.size(0) - cur_id)
            state_var = Variable(states[cur_id:cur_id+cur_batch_size])
            action_var = Variable(actions[cur_id:cur_id+cur_batch_size])
            advantages_var = Variable(advantages[cur_id:cur_id+cur_batch_size])

            #compute old and new action probabilities
            action_means, action_log_stds, action_stds = policy_net(state_var)
            log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

            action_means_old, action_log_stds_old, action_stds_old = old_policy_net(state_var)
            log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

            # updating the critic...
            #updating the value function
            value_optim.zero_grad()
            value_var = value_net(state_var)
            value_loss = (value_var - targets[cur_id:cur_id + cur_batch_size]).pow(2.).mean()
            value_loss.backward()
            value_optim.step()

            #update policy network use ppo
            policy_optim.zero_grad()
            ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
            surr1 = ratio * advantages_var[:,0]
            surr2 = torch.clamp(ratio, 1.0 - new_clip, 1.0 + new_clip) * advantages_var[:,0]
            policy_surr = -torch.min(surr1, surr2).mean()
            policy_surr.backward()
            torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
            policy_optim.step()

            # set new starting point for batch
            cur_id += cur_batch_size


# fix the loading of the values first
for ep in range(total_iterations):
    batch, reward_batch = collect_samples(policy_net, min_batch_size)
    new_lr = learning_rate*max(1.0 - float(ep)/total_iterations, 0)
    new_clip = clip_epsilon*max(1.0 - float(ep)/total_iterations, 0)
    update_parameters(batch, new_lr, new_clip)

    if ep % 10 == 0:
        print('Episode {}\treward_batch {}'.format(ep, reward_batch))

    if ep % 100 == 0:
        pcpu = policy_net.cpu()
        vcpu = value_net.cpu()
        pickle.dump((pcpu, vcpu), open('learned_models/'+ env_name+ str(min_batch_size) + '_saved_networks.p', 'wb'))
        policy_net.cuda()
        value_net.cuda()
