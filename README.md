# Decision Modeling Using Reinforcement Learning
## Part 1: Decision Graph Traversal Using Risky Q-Learning
Employs graph theory to model adversary moves within a strategic environment. An improved Q-learning algorithm (Risky Q-learning) is imployed that takes into account rewards, risks, and different consequences to those risks associated with each move in the graph and chooses the most optimal path. Below is the decision making graph used:
![Decision Making Graph](https://github.com/alanali/Reinforcement-Graph-Traversal/blob/main/Decision%20Graphs/Full_graph.jpg?raw=true)


### Modified Algorithm Performance vs Original Algorithm Performance
![Modified Algorithm Performance](https://github.com/alanali/Reinforcement-Graph-Traversal/blob/main/Graphs/Comparisons/-0.5_comp.png?raw=true)

## Part 2: Risk Modification to Influence Adversary Decision Making
An algorithm is implemented to negatively impact adversary decision making by modifiying risk values in the decision making graph to reduce the adversary's average reward and success rate of their decisions to reach an end goal.

Disclaimer: Works best with the consequence value set to 0

### Average Reward With Risk Modification vs No Risk Modification
![Risk Modification Reward](https://github.com/alanali/Reinforcement-Graph-Traversal/blob/main/Graphs/Risk%20Modification/0.0_reward_mod.png?raw=true)

### Average Success Rate with Risk Modification vs No Risk Modification
![Risk Modification Success](https://github.com/alanali/Reinforcement-Graph-Traversal/blob/main/Graphs/Risk%20Modification/0.0_rate_mod.png?raw=true)
