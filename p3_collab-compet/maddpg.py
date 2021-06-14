from ddpg import *
import torch
import numpy as np
import torch.nn.functional as F
device = 'cpu'

f_nptorch = lambda x: torch.from_numpy(x).float()

class MADDPG:
    def __init__(self, discount_factor=0.99, tau=0.008, buffer_size=10e6, batch_size=128, update_every=1, update_steps=1, noise=1, noise_reduction=0.99, lr_actor=0.001, lr_critic=0.001):
        super(MADDPG, self).__init__()
        self.maddpg_agent = [DDPGAgent(24, 64, 64, 2, 48, 64, 64, lr_actor, lr_critic), 
                             DDPGAgent(24, 64, 64, 2, 48, 64, 64, lr_actor, lr_critic)]
        self.num_agents = len(self.maddpg_agent)
        self.discount_factor = discount_factor
        self.tau = tau
        self.buffer = ReplayBuffer(int(buffer_size))
        self.update_every = update_every
        self.update_steps = update_steps
        self.batch_size = batch_size
        self.count_iter = 0
        self.noise = noise
        self.noise_reduction = noise_reduction
        
    def act(self, observations): 
        actions = [*map(lambda x: x[0].act( f_nptorch(x[1]), self.noise ).numpy(), zip(self.maddpg_agent, observations) )]
        return actions
        
    def update(self):
        # LOCALS
        for agent_id in range(self.num_agents):
            samples = self.buffer.sample(self.batch_size)
            observations, observations_all, actions, next_observations, next_observations_all, rewards, dones = samples
            rewards = rewards[:,agent_id].unsqueeze_(1)
            dones = dones[:,agent_id].view(-1, 1)
            # Critic
            with torch.no_grad():
                next_actions = torch.cat([*map(lambda a: a[1].target_actor(next_observations[:,a[0],:]), enumerate(self.maddpg_agent) )], dim=1)         
                q_next = self.maddpg_agent[agent_id].target_critic(next_observations_all, next_actions)
                y = rewards + self.discount_factor * q_next * (1 - dones)
            q = self.maddpg_agent[agent_id].critic(observations_all, actions)
            loss = F.mse_loss(q, y)
            self.maddpg_agent[agent_id].critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.maddpg_agent[agent_id].critic.parameters(), 1)
            self.maddpg_agent[agent_id].critic_optimizer.step()
            ## Actor 
            f_detach = lambda x,i: x.detach() if i!=agent_id else x
            local_actions = torch.cat([*map(lambda a: f_detach(a[1].actor(next_observations[:,a[0],:]),a[0]), enumerate(self.maddpg_agent))], dim=1)      
            q = self.maddpg_agent[agent_id].critic(observations_all, local_actions)
            loss = -q.mean()
            self.maddpg_agent[agent_id].actor_optimizer.zero_grad()
            loss.backward()
            self.maddpg_agent[agent_id].actor_optimizer.step()
        # TARGETS 
        for agent in self.maddpg_agent:
            soft_update(agent.target_actor, agent.actor, self.tau)
            soft_update(agent.target_critic, agent.critic, self.tau)

    def step(self, transition):
        self.buffer.push(transition)
        self.count_iter+=1
        self.noise*=self.noise_reduction
        if len(self.buffer) > self.batch_size:
            if self.count_iter%self.update_every==0: 
                for _ in range(self.update_steps):
                    self.update()

    def reset_noises(self):
        for agent in self.maddpg_agent:
            agent.reset_noise()                
                         
    def save_weights(self, path):
        for i,agent in enumerate(self.maddpg_agent):
            torch.save(agent.actor.state_dict(), path + f'/checkpoint_actor_{str(i)}.pth')
            torch.save(agent.critic.state_dict(), path + f'/checkpoint_critic_{str(i)}.pth')
            
    def load_weights(self, path):
        for i in range(self.num_agents):
            self.maddpg_agent[i].actor.load_state_dict(torch.load(path + f'/checkpoint_actor_{str(i)}.pth', map_location=device))
        