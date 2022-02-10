from system import dodgeball_agents
import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
import csv

f = open("rewards.txt",'w')
writer = csv.writer(f)

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    scenario = 'simpenv'
    env = dodgeball_agents("/home/brabeem/Documents/deepLearning/builds/envs/too-simple/small_map_touch_flag.x86_64")
    env.set_env()
    n_agents = env.n
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.agent_obs_size)
    critic_dims = sum(actor_dims)

    n_actions = 5
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=128, fc2=64,  
                           alpha=0.01, beta=0.001, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(50000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    N_GAMES = 10000
    MAX_STEPS = 5000
    total_steps = 0
    score_history = []
    evaluate = False

    
    maddpg_agents.load_checkpoint()
    
    for i in range(N_GAMES):
        obs = env.reset()
        score = np.array([0,0,0,0])
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done = env.step(actions)
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents
            memory.store_transition(obs, state, actions, reward, obs_, state_, done)
            if 2.0 in reward:
                print("experience multiplied")
                for j in range(4000):
                    memory.store_transition(obs, state, actions, reward, obs_, state_, done)
            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_
            score = score + np.array(reward)
            total_steps += 1
            episode_step += 1
        score_history.append(score)
        latest_score_history = np.array(score_history[-5:])
        avg_score = np.mean(latest_score_history,axis=0,keepdims=False)
        writer.writerow(avg_score)

        if not evaluate:
            if i%5 == 0:
                maddpg_agents.save_checkpoint()
                print("\r average_score:{} in episode {}".format(avg_score,i),end=" ")