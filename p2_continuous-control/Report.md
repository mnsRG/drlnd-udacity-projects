### Method

- The current implementation solves the version 2 of the project (20 parallel agents).
- A DDPG architecture is used, following the benchmark described in the project's description.
- The Actor network is a 2-Layer neural network of 258 and 128 nodes each, followed by a tanh function
- The Critic network is a 2-Layer neural network of 258 and 128 nodes each, followed by a relu function.
- For each episode the agent is trained for the <b>maximum timesteps</b> until one of the 20 parallel agents is done.
- The networks are updated <b>UPDATE_STEPS = 10</b> times every <b>UPDATE_FREQ = 20</b> episodes.
- The replay buffer size is set to <b>BUFFER_SIZE = 100000</b>.
- The minibatch size is <b>BATCH_SIZE = 128</b>.
- A discount factor of <b>GAMMA = 0.99</b> is used.
- The target parameters have a soft update of <b>TAU = 1e-3</b> .
- The learning rate of the actor is set to <b>LR_ACTOR = 1e-4</b>.
- The learning rate of the critic is set to <b>LR_CRITIC = 1e-4</b>.
- The critic's optimiser weight decay is set to <b>WEIGHT_DECAY = 0</b>.
- The training stops when the 20 parallel agents achieve an average score > <b>30.0</b> for 100 consecutive episodes.

### Plot

###### The agent solves the environment (average score > 30.0) in 135 episodes.
<br>

![plot](./plots/DDPG.png)

*Episode 100	- Average Score: 17.71* \
*Episode 135	- Average Score: 30.08*

### Future work
The agent's performance could be improved by using one of the following architectures:
1. Trust Region Policy Optimisation (TRPO)
2. Truncated Natural Policy Gradient (TNPG)
