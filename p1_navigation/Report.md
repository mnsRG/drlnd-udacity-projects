### Method

- The initial implementation was done with a <b>vanilla DQN</b> network.
- Later, <b>DQN</b> with <b>Double Q-learning</b> was tested and showed faster convergence.
- For each episode the agent is trained for a maximum of 1000 timesteps.
- E-greedy action selection is used with a 0.01 minimum value.
- The Q neural network has two fully-connected hidden layers of 64 units each, with relu activation and an output layer of 4 units (size of action values).
- The training stops when the agent achieves a score >= <b>15.0</b> for 100 consecutive episodes.

### Plot

###### The agent solves the environment (score > 13.0) in 500 episodes.
<br>

![plot](./plots/DDQN.png)

*Episode 100 - Average Score: 1.80* \
*Episode 200 - Average Score: 6.50* \
*Episode 300 - Average Score: 9.75* \
*Episode 400 - Average Score: 11.97* \
*Episode 500 - Average Score: 13.26* \
*Episode 600 - Average Score: 13.88* \
*Episode 700 - Average Score: 14.60* \
*Episode 800 - Average Score: 15.38*


### Future work
Some improvements that expect to improve the training process involve:
1. Implementing Prioritized Experience Replay.
2. Using a Dueling DQN architecture.
