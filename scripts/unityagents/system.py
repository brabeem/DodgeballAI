from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple,  ActionSpec, BehaviorSpec, DecisionStep
import numpy as np
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
class dodgeball_agents:
    def __init__(self,file_name):
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=20)
        self.side_channels = [channel]
        self.file_name = file_name
        self.worker_id = 5
        self.seed = 4
        self.env=None
        self.nbr_agent=2
        self.n = self.nbr_agent * 2
        self.spec=None
        self.agent_obs_size = 504
        self.num_envs = 1
        self.num_time_stacks = 1
        self.decision_steps = []
        self.terminal_steps = []
        self.agent_ids=([0, 1],[2, 3])
        
    ##return the environment from the file
    def set_env(self):
        self.env=UnityEnvironment(file_name=self.file_name,worker_id=self.worker_id, seed=self.worker_id, side_channels=self.side_channels)
        self.env.reset()
        self.spec=self.team_spec() 
        d0,t0 = self.env.get_steps(self.get_teamName(teamId = 0))
        d1,t1 = self.env.get_steps(self.get_teamName(teamId = 1))
        self.decision_steps.insert(0,d0)
        self.decision_steps.insert(1,d1)
        self.terminal_steps.insert(0,t0)
        self.terminal_steps.insert(1,t1)
        assert len(self.decision_steps[0]) == len(self.decision_steps[1])
        self.nbr_agent=len(self.decision_steps[0])
        if self.num_envs > 1:
            self.agent_ids = ([0, 19, 32],[37, 51, 68]) #(purple, blue) 
        else:
            self.agent_ids = ([0, 1],[2, 3])
    
    ##specify the behaviour name for the corresponding team,here in this game id is either 0 or 1
    def get_teamName(self,teamId=0):
        assert teamId in [0,1]
        return list(self.env.behavior_specs)[teamId]

    ## define the specification of the observation and actions of the environment
    def team_spec(self):
        return self.env.behavior_specs[self.get_teamName()]

    ## continous and descrete actions
    def action_size(self):
        return self.spec.action_spec
    
    ## observation size in [(3, 8), (738,), (252,), (36,), (378,), (20,)] format
    def obs_size(self):
        return [self.spec.observation_specs[i].shape for i in range(len(self.spec.observation_specs))]

    #close the environment
    def close(self):
        self.env.close()

    ## move the environment to the next step
    def set_step(self):
        self.env.step()
        self.decision_steps[0],self.terminal_steps[0] = self.env.get_steps(self.get_teamName(teamId=0))
        self.decision_steps[1],self.terminal_steps[1] = self.env.get_steps(self.get_teamName(teamId=1))

    ## set the action for each agent of respective team
    def set_action_for_agent(self,teamId,agentId,act_continuous,act_discrete):
        assert type(act_continuous) == np.ndarray and type(act_discrete) == np.ndarray
        assert act_continuous.shape[1] == self.spec.action_spec.continuous_size and act_continuous.shape[0] == 1 \
                and act_discrete.shape[1] == self.spec.action_spec.discrete_size and act_discrete.shape[0] == 1
        action_tuple = ActionTuple()
        action_tuple.add_continuous(act_continuous)
        action_tuple.add_discrete(act_discrete)
        self.env.set_action_for_agent(self.get_teamName(teamId),self.agent_ids[teamId][agentId], action_tuple)

    ##set the action for all agents of the repective team
    def set_action_for_team(self,teamId,act_continuous,act_discrete):
        assert type(act_continuous) == np.ndarray and type(act_discrete) == np.ndarray
        assert act_continuous.shape[1] == self.spec.action_spec.continuous_size and act_continuous.shape[0] == self.nbr_agent \
                and act_discrete.shape[1] == self.spec.action_spec.discrete_size and act_discrete.shape[0] == self.nbr_agent
        action_tuple = ActionTuple()
        action_tuple.add_continuous(act_continuous)
        action_tuple.add_discrete(act_discrete)
        self.env.set_actions(self.get_teamName(teamId),action_tuple)
    
    
    
    ##returns decision step for single agent from decision steps, team_id (0 or 1) and agent_index(0 or 1 or 2)
    def get_agent_decision_step(self,decision_steps, team_id, agent_index):
        assert team_id in [0, 1]
        assert agent_index in range(self.nbr_agent)
        return decision_steps[self.agent_ids[team_id][agent_index]]
        
    
    ##given a decision step corresponding to a particular agent, return the observation as a long 1 dimensional numpy array
    def get_agent_obs_with_n_stacks(self, decision_step, num_time_stacks=1):
        #TODO: ainitialize with a big enough result instead of repetitive concatenation
        assert num_time_stacks >= 1
        obs = decision_step.obs
        # print(obs[0].shape) ## (246,)
        # print(obs[1].shape) ## (84,)
        # print(obs[2].shape) ## (2,8)
        # print(obs[3].shape) ## (12,)
        # print(obs[4].shape) ## (20,)
        # print(obs[5].shape) ## (126,)
        result = np.concatenate((obs[0],obs[1]))
        result = np.concatenate((result,obs[2].reshape((-1,))))
        for i in range(3, 6):
            result = np.concatenate((result, obs[i]))
        assert result.shape[0] == 504
        return result
    

    ##returns agent observation from team decision_steps
    def get_agent_obs_from_decision_steps(self, decision_steps, team_id, agent_index, num_time_stacks=1):
        decision_step = self.get_agent_decision_step(decision_steps, team_id, agent_index)
        return self.get_agent_obs_with_n_stacks(decision_step, num_time_stacks)
        
        
    ##returns concatenated team observation from team decision_steps
    def get_team_obs_from_decision_steps(self, decision_steps, team_id, num_time_stacks=1):
        team_obs = np.zeros(shape=(self.nbr_agent*self.agent_obs_size,))
        for idx in range(self.nbr_agent):
            team_obs[self.agent_obs_size*idx:self.agent_obs_size*(idx+1)] = self.get_agent_obs_from_decision_steps(decision_steps, team_id, idx, num_time_stacks)
        return team_obs
            
    ##returns agent reward ##    
    def reward(self,team_id,agent_index):
        if self.agent_ids[team_id][agent_index] in self.decision_steps[team_id].agent_id:
            reward = self.decision_steps[team_id].__getitem__(self.agent_ids[team_id][agent_index]).reward
            grp_reward = self.decision_steps[team_id].__getitem__(self.agent_ids[team_id][agent_index]).group_reward
        if self.agent_ids[team_id][agent_index] in self.terminal_steps[team_id].agent_id:
            reward = self.terminal_steps[team_id].__getitem__(self.agent_ids[team_id][agent_index]).reward
            grp_reward = self.terminal_steps[team_id].__getitem__(self.agent_ids[team_id][agent_index]).group_reward
        return reward + grp_reward
    
    ##returns done##
    def terminal(self,team_id,agent_index):
        if self.agent_ids[team_id][agent_index] in self.decision_steps[team_id].agent_id:
            done = False
        if self.agent_ids[team_id][agent_index] in self.terminal_steps[team_id].agent_id:
            done = True
        return done 
   
    ##get all agent obs as a list where each element in the list corresponds to an agent's observation##
    def get_all_agent_obs(self):
        obs = []
        for teamid in range(2):
            for agentIndex in range(2):
                obs.append(self.get_agent_obs_from_decision_steps(self.decision_steps[teamid],teamid,agentIndex,1))
        return obs

    ##reset the environment like gym##
    def reset(self):
        self.env.reset()
        self.decision_steps[0],self.terminal_steps[0] = self.env.get_steps(self.get_teamName(teamId = 0))
        self.decision_steps[1],self.terminal_steps[1] = self.env.get_steps(self.get_teamName(teamId = 1))
        return self.get_all_agent_obs()
    
    ##puting the above set_action_for_agent in a convinient way##
    def set_action_for_agent_(self,teamId,agentInd,action_tuple):
        self.set_action_for_agent(teamId=teamId,agentId=agentInd,act_continuous=action_tuple.continuous,act_discrete=action_tuple.discrete)
    
    ##get rewards as a list of reward of each agent in the game##
    def get_all_agent_reward(self):
        rewards = []
        for teamId in range(2):
            for agentInd in range(2):
                rewards.append(self.reward(teamId,agentInd))
        return rewards
    
    
    ##get all agent dones as a list of each agent done##
    def get_all_agent_done(self):
        dones = []
        for teamId in range(2):
            for agentInd in range(2):
                dones.append(self.terminal(teamId,agentInd))
        return dones
    
    ##converts numpy list into list of action tuples##
    def numpy_list_to_action_tuple_list(self, numpy_list):
        num_continuous = 3
        action_tuple_list = []
        for element in numpy_list:
            action_tuple_continuous =  element[:num_continuous]
            action_tuple_discrete = element[num_continuous:]
            action_tuple_list.append(ActionTuple(np.expand_dims(action_tuple_continuous,0),np.expand_dims(action_tuple_discrete,0)))
        return action_tuple_list

    ##step equivalent of gym environment##
    ##expects actions to be the list of numpy arrays and [cont,cont,cont,dis,dis] is a array of continuous and 
    # discrete actions##
    def step(self,actions):
        action_idx = 0
        ##bring the list of action tuples from the list of numpy arrays##
        actions = self.numpy_list_to_action_tuple_list(actions)
        ##set action for all agents##
        for teamId in range(2):
            for agentInd in range(2):
                self.set_action_for_agent_(teamId,agentInd,actions[action_idx])
                action_idx += 1
        
        ##step insimulation##
        self.env.step()
        
        ##get the new decision and terminal steps##
        self.decision_steps[0],self.terminal_steps[0] = self.env.get_steps(self.get_teamName(teamId = 0))
        self.decision_steps[1],self.terminal_steps[1] = self.env.get_steps(self.get_teamName(teamId = 1))
        
        ##recently added to resolve err##
        dones = self.get_all_agent_done()
        if any(dones[0:2]):
            self.decision_steps[0] = self.terminal_steps[0]
        if any(dones[2:4]):
            self.decision_steps[1] = self.terminal_steps[1]

        ##get next_states ,rewards and dones from updated decision and terminal steps##
        next_states  = self.get_all_agent_obs()
        rewards = self.get_all_agent_reward()
        # dones = self.get_all_agent_done()
        
        return next_states,rewards,dones

##return random action for checking if functionalities are working
##to replicate this kind of actions from network we have to transpose the network output##
    def random_action(self):
        actions = []
        for i in range(6):
            a = np.random.random(5)
            a = np.expand_dims(a,axis=0)
            ##format(1,5)
            actions.append(a)
        return actions

    
        
    
    
            
            
        


