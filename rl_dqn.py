from collections import namedtuple, deque
from itertools import count

import tqdm
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import random
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from torch.utils.tensorboard import SummaryWriter



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                nn.init.constant_(m.bias, 0.1)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

class RLModel:
    
    def __init__(self,env,test_env,device,addscore_emb=False,log_path='rl_log',memory_size=10000):
        self.BATCH_SIZE = 128
        self.TAU = 0.005
        self.LR = 1e-4
        self.GAMMA = 0.99
        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, info = env.reset()
        n_observations = len(state)
        if addscore_emb:
            n_observations += 3

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.policy_net.initialize_weights()
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(memory_size)
        self.env = env
        self.test_env = test_env
        self.device = device
        
        self.log_path = log_path
        self.writer = SummaryWriter(log_path)
        
        self.steps_done = 0
        self.metrics = {'reward':-np.inf}
        
        
    def caculate_threshold(self):
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 10000

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        return eps_threshold
        
        
    def select_action(self, state, eps_threshold=None):
        # global steps_done
        sample = random.random()
        if  eps_threshold is None:
            eps_threshold = self.caculate_threshold()
        if sample >= eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1), eps_threshold
        else:
            return torch.tensor([[self.env.action_space.sample()]], device= self.device, dtype=torch.long), eps_threshold


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes=3000,start_episodes=0,evaluate=True,debug=False,eval_step=500):
        self.env.set_debug(debug)
        self.test_env.set_debug(debug)

        return_list = []
        episodes_list = []
        test_episodes_list, train_episodes_list = None, None
        
        for i_episode in tqdm.tqdm(range(start_episodes,start_episodes+num_episodes),desc='RL Training'):
            # Initialize the environment and get it's state
            returns = 0
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            self.writer.add_scalar('train/returns_episode', returns, i_episode)
            if evaluate and i_episode%eval_step==0:
                test_metrics = self.evaluate(self.test_env,eva_tag='eva test:')
                test_episodes_list, test_Rewards, test_accurate_match_rate = \
                test_metrics['episodes_list'], test_metrics['rewards'], test_metrics['accurate_match_rate']
                self.writer.add_scalar('test/Rewards_all', test_Rewards, self.steps_done)
                self.writer.add_scalar('test/accurate_match_rate', test_accurate_match_rate, i_episode)
                self.save_result(i_episode, test_Rewards)
                
                train_metrics = self.evaluate(self.env,eva_tag='eva train:')
                train_episodes_list, train_Rewards, train_accurate_match_rate = \
                train_metrics['episodes_list'], train_metrics['rewards'], train_metrics['accurate_match_rate']
                self.writer.add_scalar('train/Rewards_all', train_Rewards, self.steps_done)
                self.writer.add_scalar('train/accurate_match_rate', train_accurate_match_rate, i_episode)
                print(i_episode)
                for key in ['rewards','accurate_match_rate','find_mean_length','find_subs_rate']:
                    print( key,test_metrics[key],train_metrics[key])

            for t in count():
                action, eps_threshold = self.select_action(state)
                self.writer.add_scalar('eps_threshold', eps_threshold, self.steps_done)
                self.steps_done += 1

                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device= self.device)
                returns += reward
                done = terminated or truncated
                # if terminated:
                #     next_state = None
                # else:
                #     next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
                # Move to the next state
                state = next_state
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                if done:
                    return_list.append(returns)
                    episodes_list.append(copy.copy(self.env.history))
                    break
            
#                 print(f"\n {i_episode} test reward：{test_Rewards}", f"accurate_rate: {round(test_accurate_match_rate,3)}", test_metrics['find_mean_length'],test_metrics['find_subs_rate'])
                
#                 print(f"train reward:{train_Rewards}", f"accurate_rate: {round(train_accurate_match_rate,3)}",train_metrics['average_length'],train_metrics['find_subs_rate'])

        return test_episodes_list, train_episodes_list
    
    def save_policy_model(self):
        with open(f'{self.log_path}/best_policy.pkl','wb') as f:
            policy_net_state_dict = self.policy_net.state_dict()
            torch.save(policy_net_state_dict,f)

            
    
    def save_result(self,i_episode, reward):
        if reward > self.metrics['reward']:
            self.save_policy_model()
            self.metrics['reward'] = reward
            with open(f'{self.log_path}/best_policy_info.txt','a',encoding='utf8') as f:
                f.write(f"{i_episode},{reward}\n")
    
    def evaluate(self,eva_env,size=None,eva_tag='',eps_threshold=0,set_action=None):
        Rewards = 0
        episodes_list = []
        find_subs = []
        history_lengths = []
        if size is None:
            size = len(eva_env.dataset.contexts)
        for context_id in tqdm.tqdm(eva_env.dataset.context_ids[:size],desc=eva_tag,mininterval=30):
            state, info = eva_env.reset(context_id)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action, _ = self.select_action(state,eps_threshold=eps_threshold)
                action = action.item()
                if set_action is not None:
                    action = set_action
                observation, reward, terminated, truncated, _ = eva_env.step(action)
                done = terminated or truncated
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                # Move to the next state
                state = next_state
                Rewards += reward
                if done:
                    episodes_list.append(copy.copy(eva_env.history))
                    # plot_durations()
                    break
            is_find = eva_env.history[-1].get('find_subs')
            find_subs.append(bool(is_find))
            history_lengths.append(len(eva_env.history))
        
        accurate_match = []
        
        for eps in episodes_list:
            for i,utter in enumerate(eps):
                if 'is_right_action' in utter:
                    accurate_match.append(utter['is_right_action'])
        accurate_match_rate = np.array(accurate_match).sum() / len(accurate_match)
        find_subs_rate = np.array(find_subs).sum() / len(accurate_match)
        
        find_mean_lengths =[]
        not_find_mean_lengths = []
        for is_find, average_length in zip(find_subs,history_lengths):
            if is_find==1:
                find_mean_lengths.append(average_length)
            else:
                not_find_mean_lengths.append(average_length)
                
        find_mean_length = np.mean(find_mean_lengths)
        not_find_mean_length = np.mean(not_find_mean_lengths)
        
        metrics = dict(episodes_list=episodes_list,
                       rewards=Rewards,
                       accurate_match_rate=accurate_match_rate,
                      find_subs_rate=find_subs_rate,
                       find_mean_length=find_mean_length,
                       not_find_mean_length=not_find_mean_length
                      )
        return metrics
