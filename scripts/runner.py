# from tqdm import tqdm
# from agent import Agent
# from common.replay_buffer import Buffer
# import torch
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from maddpg.agents import dodgeball_agents
# from collections import deque
# class Runner:
#     def __init__(self, args, env):
#         self.args = args
#         self.noise = args.noise_rate
#         self.epsilon = args.epsilon
#         self.episode_limit = args.max_episode_len
#         self.env = env
#         self.agents = self._init_agents()
#         self.buffer = Buffer(args)
#         self.avg_returns_test={'team_purple':[],'team_blue':[]}
#         self.avg_returns_train={'team_purple':[],'team_blue':[]}
#         self.count_ep=0
#         self.scores_deque = {'team_purple':deque(maxlen=5),'team_blue':deque(maxlen=5)}
#         self.save_path = self.args.save_dir + '/' + self.args.scenario_name
#         if not os.path.exists(self.save_path):
#             os.makedirs(self.save_path)

#     def _init_agents(self):
#         agents = []
#         for i in range(self.args.n_agents):
#             agent = Agent(i, self.args)
#             agents.append(agent)
#         return agents

#     def select_action_for_opponent(self,agent_id):
#             a=np.random.uniform(-1,1, self.args.action_shape[agent_id])*0.5
#             #a=np.zeros(self.args.action_shape[agent_id])
#             a[:self.args.continuous_action_space]= self.args.high_action*a[:self.args.continuous_action_space]
#             a[self.args.continuous_action_space:]=(np.abs(a[self.args.continuous_action_space:])>0.2)*1
#             return a

#     def run(self):
#         returns = {'team_purple':[],'team_blue':[]}
#         n=self.args.n_agents
#         for i in range(self.episode_limit+1):
#             s,r,gr,d = self.env.reset()
#             # for agent in self.agents:
#             #     agent.noise.reset()
#             rewards = {'team_purple':0,'team_blue':0}
#             for time_step in tqdm(range(self.args.time_steps)):
#                 a = []
#                 a_opponent = []
#                 with torch.no_grad():
#                     for agent_id, agent in enumerate(self.agents):
#                         action = agent.select_action(s[agent_id], self.noise, self.epsilon)
#                         a.append(action)
#                         a_opponent.append(self.select_action_for_opponent(agent_id))
#                         # if agent_id==1:
#                         #     print(s[agent_id])
#                 for j in range(self.args.n_agents):
#                     a.append(a_opponent[j])
#                 s_next, r,gr, done= self.env.step(a)
#                 r=list((np.array(r)+np.array(gr)))
#                 rewards['team_purple'] += np.sum(r[:n])
#                 rewards['team_blue'] += np.sum(r[n:])
#                 # if (np.sum(r[:3])>0.0001):
#                 #     for _ in range(50):
#                 self.buffer.store_episode(s[:n], a[:n], r[:n], s_next[:n],done[:n])
#                 s = s_next
#                 if self.buffer.current_size >= self.args.batch_size and time_step%self.args.learn_rate==0:
#                     experiences = self.buffer.sample(self.args.batch_size)
#                     for agent in self.agents:
#                         other_agents = self.agents.copy()
#                         other_agents.remove(agent)
#                         agent.learn(experiences, other_agents)
#                 # if time_step > 0 and time_step % self.args.evaluate_rate == 0:
#                 #     self.evaluate()
#                 #     self.plot_graph( self.avg_returns_test['team_blue'],list( self.avg_returns_test.keys())[0],method='test')
#                 #     self.plot_graph( self.avg_returns_test['team_purple'],list( self.avg_returns_test.keys())[1],method='test')
#                 if(any(done)==True):
#                     break
#             self.noise = max(0.01, self.noise - 0.0002)
#             self.epsilon = max(0.01, self.epsilon - 0.0002)
#             #returns['team_blue'].append(rewards['team_blue'])
#             #returns['team_purple'].append(rewards['team_purple'])
#             self.scores_deque['team_blue'].append(rewards['team_blue'])
#             self.scores_deque['team_purple'].append(rewards['team_purple'])
#             print('team blue avg Returns is', np.mean(self.scores_deque['team_blue']))
#             print('team purple avg Returns is',np.mean(self.scores_deque['team_purple']))  
#             self.avg_returns_train['team_blue'].append(np.mean(self.scores_deque['team_blue']))
#             self.avg_returns_train['team_purple'].append(np.mean(self.scores_deque['team_purple']))
#             self.plot_graph(self.avg_returns_train,method='train')
#             # if(np.mean(self.scores_deque['team_purple'])>3 and i >100):
#             #     break
#         return self.avg_returns_train
    
#     def plot_graph(self,avg_returns,method=None):
#         plt.figure()
#         plt.plot(range(len(avg_returns['team_blue'])),avg_returns['team_blue'])
#         plt.plot(range(len(avg_returns['team_purple'])),avg_returns['team_purple'])
#         plt.xlabel('episode')
#         plt.ylabel('average returns')
#         plt.legend(["blue_reward","purple_reward"])
#         if method=='test':
#             plt.savefig(self.save_path + '/' + 'test_plt.png' , format='png')
#         else:
#             plt.savefig(self.save_path + '/' + 'train_plt.png' , format='png')
           
#         # np.save(self.save_path + '/' + name + ('%d' % self.count_ep) + '_team_blue_returns.pkl',self.avg_returns['team_blue'])
#         # np.save(self.save_path + '/' + name + ('%d' % self.count_ep) + '_team_purple_returns.pkl',self.avg_returns['team_purple'])

#     def evaluate(self):
#         n=self.args.n_agents
#         returns = {'team_purple':[],'team_blue':[]}
#         for episode in range(self.args.evaluate_episodes):
#                     # reset the environment
#             rewards = {'team_purple':0,'team_blue':0}
#             s,r,gr,d = self.env.reset()
#             for time_step in range(self.args.evaluate_episode_len):
#                 a = []
#                 a_opponent = []
#                 with torch.no_grad():
#                     for agent_id, agent in enumerate(self.agents):
#                         action = agent.select_action(s[agent_id],0,0)
#                         a.append(action)
#                         a_opponent.append(self.select_action_for_opponent(agent_id))
#                 for j in range(self.args.n_agents):
#                     a.append(a_opponent[j])
#                 s_next, r,gr, done= self.env.step(a)
#                 r=list((np.array(r))+np.array(gr))
#                 rewards['team_purple'] += np.mean(r[:n])
#                 rewards['team_blue'] += np.mean(r[n:])
#                 s = s_next[:n]
#                 if(any(done)==True):
#                     break
#             returns['team_blue'].append(rewards['team_blue'])
#             returns['team_purple'].append(rewards['team_purple'])
#             print('team blue Returns is', rewards['team_blue'])
#             print('team purple Returns is', rewards['team_purple'])
#         self.avg_returns_test['team_blue'].append(np.mean(returns['team_blue']))
#         self.avg_returns_test['team_purple'].append(np.mean(returns['team_purple']))


from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from maddpg.agents import dodgeball_agents
from collections import deque
import csv
import numpy as np
from random import sample

# f_blue = open("blue_returns.txt", 'a')
# f_purple = open("purple_returns.txt", 'a')
# f_elo = open("elo_rating", 'a')

# writer_blue = csv.writer(f_blue)
# writer_purple = csv.writer(f_purple)
# writer_elo = csv.writer(f_elo)

f = open("results.txt", 'a')
writer = csv.writer(f)
class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.buffer = Buffer(args)
        self.avg_returns_test={'team_purple':[],'team_blue':[], 'sum_reward':[]}
        self.avg_returns_train={'team_purple':[],'team_blue':[], 'sum_reward':[]}
        self.elo_ratings = [1200, 1200]
        self.count_ep=0
        self.scores_deque = {'team_purple':deque(maxlen=5),'team_blue':deque(maxlen=5)}
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        self.network_bank = deque(maxlen=self.args.size_netbank)
        self.opponent_networks = None
        self.agents = [[], []]
        self.last_opponent_index = -1
        self._init_agents()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        self.opponent_actors=[]
        for team_id in range(2):
            for i in range(self.args.n_learning_agents):
                self.agents[team_id].append(Agent(team_id*self.args.n_learning_agents + i, self.args))

    def swap_opponent_team(self):
        select_latest = np.random.binomial(1, p=self.args.p_select_latest)
        if select_latest or len(self.network_bank) == 1:
            self.opponent_networks = self.network_bank[self.last_opponent_index]
        elif len(self.network_bank) == 0:
            return
        else:
            self.last_opponent_index = np.random.randint(0, len(self.network_bank)-1)
            self.opponent_networks = self.network_bank[self.last_opponent_index]
            self.elo_ratings[1] = 1200
        
        for agent_id in range(self.args.n_learning_agents):
            self.agents[1][agent_id].load_actor_params(self.opponent_networks[agent_id])

    def update_elo_rating(self, results):
        expected = [None, None]
        rdiff_0 = self.elo_ratings[1] - self.elo_ratings[0]
        rdiff = [rdiff_0, -rdiff_0]
        for index in range(len(self.elo_ratings)):
            expected[index] = 1/(1 + 10**(rdiff[index]/400))
            self.elo_ratings[index] += 5*(results[index] - expected[index])

    def run(self):
        n=self.args.n_learning_agents
        time_step=0
        for i in range(self.episode_limit+1):
            # self.avg_returns_train={'team_purple':[],'team_blue':[], 'sum_reward':[]}
            print("Episode: ", i, " Steps: ", time_step)
            s,r,gr,d = self.env.reset()
            # for agent in self.agents:
            #     #TODO load state dict to opponent teamagent.noise.reset()
            rewards = {'team_purple':0,'team_blue':0}
            while True:
                
                a = []
                with torch.no_grad():
                    for team_id in range(2):
                        for agent_id, agent in enumerate(self.agents[team_id]):
                            a.append(agent.select_action(s[team_id*self.args.n_learning_agents+agent_id], self.noise, self.epsilon))
                s_next, r,gr, done= self.env.step(a)
                r=list((np.array(r)+np.array(gr)))
                rewards['team_purple'] += np.sum(r[:2])
                rewards['team_blue'] += np.sum(r[2:])
                # if (np.sum(r[:3])>0.0001):
                #     for _ in range(50):
                self.buffer.store_episode(s[:n], a[:n], r[:n], s_next[:n],done[:n])
                s = s_next
                if self.buffer.current_size >= self.args.batch_size and time_step%self.args.learn_rate==0:
                    experiences = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents[0]:
                        other_agents = self.agents[0].copy()
                        other_agents.remove(agent)
                        agent.learn(experiences, other_agents)
                
                
                if time_step % self.args.save_team==0:
                    print("Saving network to network bank")
                    self.network_bank.append([self.agents[0][idx].get_actor_params() for idx in range(self.args.n_learning_agents)])
                
                if time_step % self.args.swap_team==0:
                    print("Swapping opponent team")
                    self.swap_opponent_team()
                # if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                #     self.evaluate()
                #     self.plot_graph( self.avg_returns_test['team_blue'],list( self.avg_returns_test.keys())[0],method='test')
                #     self.plot_graph( self.avg_returns_test['team_purple'],list( self.avg_returns_test.keys())[1],method='test')
                time_step += 1
                if any(done):
                    #TODO: result = 1 if win, 0 if lose, 0.5 if draw
                    if r[0] >= 1:
                        result = [1,0]
                    elif r[3] >= 1:
                        result = [0,1]
                    else:  
                        result = [0.5,0.5] 
                    self.update_elo_rating(result)
                    break
            self.noise = max(0.01, self.noise - 0.004)
            self.epsilon = max(0.01, self.epsilon - 0.0002)
            #returns['team_blue'].append(rewards['team_blue'])
            #returns['team_purple'].append(rewards['team_purple'])
            self.scores_deque['team_blue'].append(rewards['team_blue'])
            self.scores_deque['team_purple'].append(rewards['team_purple'])
            # print('team blue avg Returns is', np.mean(self.scores_deque['team_blue']))
            # print('team purple avg Returns is',np.mean(self.scores_deque['team_purple']))  
            # print('team purple avg Returns is',np.mean(self.scores_deque['team_purple'])) 
            blue_avg_reward =  float(np.mean(self.scores_deque['team_blue']))
            purple_avg_reward = float(np.mean(self.scores_deque['team_purple']))
            # self.avg_returns_train['team_blue'].append(blue_avg_reward)
            # self.avg_returns_train['team_purple'].append(purple_avg_reward)
            # writer_blue.writerow(self.avg_returns_train['team_blue'])
            # writer_purple.writerow(self.avg_returns_train['team_purple'])
            # writer_elo.writerow(self.elo_ratings)
            writer.writerow([blue_avg_reward, purple_avg_reward, self.elo_ratings])
            # self.avg_returns_train['sum_reward'].append(blue_avg_reward+purple_avg_reward)
            # 
            # self.plot_graph(self.avg_returns_train,method='train')
            if i % 5 == 0:
                # f_blue.flush()
                # f_purple.flush()
                # f_elo.flush()
                for agent in self.agents[1]:
                    agent.policy.save_model()
                f.flush()
            # if(np.mean(self.scores_deque['team_purple'])>3 and i >100):
            #     break
        return self.avg_returns_train
    
    def plot_graph(self,avg_returns,method=None):
        plt.figure()
        plt.plot(range(len(avg_returns['team_blue'])),avg_returns['team_blue'])
        plt.plot(range(len(avg_returns['team_purple'])),avg_returns['team_purple'])
        # plt.plot(range(len(avg_returns['sum_reward'])),avg_returns['sum_reward'])
        plt.xlabel('episode')
        plt.ylabel('average returns')
        plt.legend(["blue_reward","purple_reward"])
        if method=='test':
            plt.savefig(self.save_path + '/' + 'test_plt.png' , format='png')
        else:
            plt.savefig(self.save_path + '/' + 'train_plt.png' , format='png')
           
        # np.save(self.save_path + '/' + name + ('%d' % self.count_ep) + '_team_blue_returns.pkl',self.avg_returns['team_blue'])
        # np.save(self.save_path + '/' + name + ('%d' % self.count_ep) + '_team_purple_returns.pkl',self.avg_returns['team_purple'])

    def evaluate(self):
        # n=self.args.n_agents
        # returns = {'team_purple':[],'team_blue':[]}
        # for episode in range(self.args.evaluate_episodes):
        #             # reset the environment
        #     rewards = {'team_purple':0,'team_blue':0}
        #     s,r,gr,d = self.env.reset()
        #     for time_step in range(self.args.evaluate_episode_len):
        #         a = []
        #         a_opponent = []
        #         with torch.no_grad():
        #             for agent_id, agent in enumerate(self.agents):
        #                 action = agent.select_action(s[agent_id],0,0)
        #                 a.append(action)
        #                 a_opponent.append(self.select_action_for_opponent(agent_id))
        #         for j in range(self.args.n_agents):
        #             a.append(a_opponent[j])
        #         s_next, r,gr, done= self.env.step(a)
        #         r=list((np.array(r))+np.array(gr))
        #         rewards['team_purple'] += np.mean(r[:n])
        #         rewards['team_blue'] += np.mean(r[n:])
        #         s = s_next[:n]
        #         if(any(done)==True):
        #             break
        #     returns['team_blue'].append(rewards['team_blue'])
        #     returns['team_purple'].append(rewards['team_purple'])
        #     print('team blue Returns is', rewards['team_blue'])
        #     print('team purple Returns is', rewards['team_purple'])
        # self.avg_returns_test['team_blue'].append(np.mean(returns['team_blue']))
        # self.avg_returns_test['team_purple'].append(np.mean(returns['team_purple']))



        n=self.args.n_agents
        time_step=0
        for i in range(self.episode_limit+1):
            self.avg_returns_train={'team_purple':[],'team_blue':[], 'sum_reward':[]}
            print("Episode: ", i, " Steps: ", time_step)
            s,r,gr,d = self.env.reset()
            # for agent in self.agents:
            #     agent.noise.reset()
            rewards = {'team_purple':0,'team_blue':0}
            while True:
                time_step += 1
                a = []
                with torch.no_grad():
                    for team_id in range(2):
                        for agent_id, agent in enumerate(self.agents[team_id]):
                            a.append(agent.select_action(s[team_id*self.args.n_learning_agents + agent_id], 0, 0))

                s_next, r,gr, done= self.env.step(a)
             
                
                s = s_next
               
                # if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                #     self.evaluate()
                #     self.plot_graph( self.avg_returns_test['team_blue'],list( self.avg_returns_test.keys())[0],method='test')
                #     self.plot_graph( self.avg_returns_test['team_purple'],list( self.avg_returns_test.keys())[1],method='test')
                if(any(done)==True):
                    break