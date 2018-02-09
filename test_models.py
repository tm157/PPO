import pickle
import numpy as np
import gym
import torch
from torch.autograd import Variable

from models.policy_network import Policy
from models.value_network import Value
from itertools import count 

max_expert_state_num = 20000
render = True
DoubleTensor = torch.DoubleTensor
Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

env = gym.make('Humanoid-v1')

model_path = 'networks_humanoid_1000Iterations.p'
policy_net, value_net = pickle.load(open(model_path, 'rb'))

expert_traj = []
num_steps = 0
for i_episode in count():
    state = env.reset()
    reward_episode = 0

    for i in range(10000):
        state_var = Variable(Tensor(state).unsqueeze(0))
        action = policy_net.select_action(state_var)[0].cpu().numpy()
        action = action.astype(np.float64)
        next_state, reward, done, _ = env.step(action)
        reward_episode += reward

        num_steps += 1
        if render:
            env.render()
        
        if done:
            break
        state = next_state
    print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))
if num_steps >= max_expert_state_num:
        break
expert_traj = np.stack(expert_traj)

