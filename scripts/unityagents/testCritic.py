import torch as T
import network as net
from system import dodgeball_agents
import csv
import numpy as np

f = open("rewards.txt",'w')
writer = csv.writer(f)

critic_filenames = ["agent_0_critic","agent_1_critic","agent_2_critic","agent_3_critic"]
actor_filenames = ["agent_0_actor","agent_1_actor","agent_2_actor","agent_3_actor"]
critic_nets = []


for _ in critic_filenames:
    critic_nets.append(net.CriticNetwork(0.001,2016,22,22,4,5,"oho","ok"))

for i,critic_net in enumerate(critic_nets):
    critic_net.load_state_dict(T.load("tmp/maddpg/smallNet/" + critic_filenames[i]))

actor_nets = []

for _ in actor_filenames:
    actor_nets.append(net.ActorNetwork(0.01,504,45,45,5,"oho","ok"))

for i,actor_net in enumerate(actor_nets):
    actor_net.load_state_dict(T.load("tmp/maddpg/smallNet/" + actor_filenames[i]))

# for actor_net in actor_nets:
#     for name,param in actor_net.named_parameters():
#         writer.writerow(param)
def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

env = dodgeball_agents("/home/brabeem/Documents/deepLearning/builds/envs/too-simple/small_map_touch_flag.x86_64")
env.set_env()
for i in range(10):
    obs = env.reset()
    done = [False] * 4
    episode_step = 0
    while not any(done):
        actions = []
        q = []
        not_torch_state = obs_list_to_state_vector(obs)
        torch_state = T.tensor(not_torch_state, dtype=T.float)
        torch_state = T.unsqueeze(torch_state,dim=0)
        for i,o in enumerate(obs):
            o = T.tensor([o], dtype=T.float)
            # print("obs:",o.shape)
            # o = T.unsqueeze(o,dim=0)
            o = actor_nets[i].forward(o)            
            # print("actions:",o.shape)
            # o = T.unsqueeze(o,dim=0)
            # print("actions_tallo:",o.shape)
            actions.append(o)
        # writer.writerow(actions)
        actions_not_list = T.cat([acts for acts in actions],dim=1)
        for j,_ in enumerate(actions):
            q_each = critic_nets[j].forward(torch_state,actions_not_list)
            q.append(float(q_each.data[0][0]))
        
        writer.writerow(q)
        actions_ = []
        for action in actions:
            actions_.append(action.squeeze().detach().numpy())
        obs,_,done = env.step(actions_)
        episode_step += 1
        if episode_step >= 2000:
            done = [True]*4
    print("number_of_steps: ",episode_step)