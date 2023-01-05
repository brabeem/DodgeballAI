Applied Multiagent Deep Deterministic policy gradient(MADDPG) to solve the dodgeball 3D game.
Created ciriculum for the agent to learn.
Applied self learning with MADDPG and achieved quite good result.

Dodgeball 3D game:
  - It is a Unity game available in UnityHub
  - The game is a capture the flag game where an agent runs to capture the flag of the opponent and 
    brings the flag of the enemy to its teritory while protecting his own flag.
MADDPG:
  -[maddpg]https://paperswithcode.com/method/maddpg
Curiculum Learning:
  -A ciriculum is created for an agent to learn i.e they are trained in simple 
   environments and as they go on learning the complexity of the environment is increased.
  -[Curiculum Learning]https://arxiv.org/abs/2101.10382
Self Learning:
  -The agents competes in symetric game with the its older version.
  -[self play]https://en.wikipedia.org/wiki/Self-play_(reinforcement_learning_technique)

Model:
1)actor network used:

![Actor](https://github.com/brabeem/DodgeballAI/blob/jhanda_uthauxa/actor.png)

2)Critic network used:
![Critic](https://github.com/brabeem/DodgeballAI/blob/jhanda_uthauxa/critic.png)


What achieved?
  -We were successful to achieve a very competitive capture the flag player but we couldnot 
   achieve considerable collaboration between players.
