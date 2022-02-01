from system import dodgeball_agents
""" # import csv
# f = open("rewards.txt",'w')
# writer = csv.writer(f)
if __name__ == "__main__":
    env = dodgeball_agents("/home/brabeem/Documents/deepLearning/builds/RewardSingEnv/sectf.x86_64")
    env.set_env()
    state = env.reset()
    discont = False
    while discont == False:
        actions  = env.random_action()
        next_state,reward,done = env.step(actions)
        # writer.writerow(reward)
        next_state = state
        discont = any(done)
    env.close() """

import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    scenario = 'ctf'
    # scenario = 'simple_adversary'
    env = dodgeball_agents("/home/brabeem/Documents/deepLearning/builds/RewardSingEnv/sectf.x86_64")
    env.set_env()
    n_agents = env.n
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.agent_obs_size)
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 5
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=128, fc2=64,  
                           alpha=0.0001, beta=0.0001, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    # PRINT_INTERVAL = 500
    N_GAMES = 10000
    MAX_STEPS = 5000
    total_steps = 0
    score_history = []
    evaluate = True
    # best_score = 0

    maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            """ if evaluate:
                env.render() """
                #time.sleep(0.1) # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                print("chiryo")
                done = [True]*n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if i%5 == 0:
                maddpg_agents.save_checkpoint()
                print("\r average_score:{} in episode {}".format(avg_score,i),end=" ")
        """ if i % PRINT_INTERVAL == 0 and i > 0:
            print('\r episode', i, 'average score {:.1f}'.format(avg_score),end=" ") """