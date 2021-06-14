from torch.optim import Adam
import torch
import numpy as np
import copy
import torch.nn as nn
from collections import deque
import random
import torch.nn.functional as f
device = 'cpu'

f_nptorch = lambda x: torch.from_numpy(x).float()

class DDPGAgent: 
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=0.001, lr_critic=0.001):
        super(DDPGAgent, self).__init__()
        self.actor = ActorNetwork(in_actor, hidden_in_actor, hidden_out_actor, out_actor).to(device)
        self.critic = CriticNetwork(in_critic, hidden_in_critic, hidden_out_critic, 1, out_actor).to(device)
        self.target_actor = ActorNetwork(in_actor, hidden_in_actor, hidden_out_actor, out_actor).to(device)
        self.target_critic = CriticNetwork(in_critic, hidden_in_critic, hidden_out_critic, 1, out_actor).to(device)
        self.noise = OUNoise(out_actor)
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=0)

    def act(self, obs, noise=0.0 ):  
        with torch.no_grad():
            self.actor.eval()
            obs = obs.to(device)
            action = self.actor(obs) 
            self.actor.train()
        action += noise*self.noise.sample()
        action = np.clip(action,-1,1)
        return action

    def reset_noise(self):
        self.noise.reset()

         
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)        

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim ):
        super(ActorNetwork,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        self.nonlin = f.relu
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)
        
    def forward(self,x):
        x = self.nonlin(self.fc1(x))
        x = self.nonlin(self.fc2(x))
        x = (self.fc3(x))
        return f.tanh(x) 
    
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, action_size ):
        super(CriticNetwork,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_in_dim)
        self.bn = nn.BatchNorm1d(hidden_in_dim+action_size*2)
        self.fc2 = nn.Linear(hidden_in_dim+action_size*2,hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        self.nonlin = f.relu
        self.reset_parameters()        
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)
        
    def forward(self,x,a):
        x = self.nonlin(self.fc1(x))
        x = torch.cat((x, a), dim=1)
        x = self.bn(x)
        x = self.nonlin(self.fc2(x))
        x = (self.fc3(x))
        return x 

    
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class OUNoise:
    def __init__(self, action_dimension, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(action_dimension)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        random_array = [random.random() for i in range(len(x))]
        dx = self.theta * (self.mu - x) + self.sigma * np.array(random_array)
        self.state = x + dx
        return self.state    
    
class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self,transition):
        self.deque.append(transition)

    def sample(self, batchsize):
        samples = random.sample(self.deque, batchsize)
        observations, actions, rewards, next_observations, dones = [*zip(*samples)]
        observations = f_nptorch(np.array(observations))
        observations_all = observations.view(batchsize,-1)
        actions = f_nptorch(np.array(actions)).view(batchsize,-1)
        next_observations = f_nptorch(np.array(next_observations))
        next_observations_all = next_observations.view(batchsize,-1)
        rewards = f_nptorch(np.array(rewards))
        dones = f_nptorch(np.array(dones))   
        return observations, observations_all, actions, next_observations, next_observations_all, rewards, dones

    def __len__(self):
        return len(self.deque)
    
