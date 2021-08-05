#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 13:21:28 2021

@author: christian brandstaetter
"""

import torch
import os
import random
import pickle
import numpy as np
from .networks import Policy_DDDQN
from .networks import Policy_QN
#from .networks import Policy_VN
from collections import deque

class Controller_Agent():
    ''' Chapter 3.4.2.2 '''
    def __init__(self, state_dim, action_dim, opt):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = opt.save_dir
        self.save_every = opt.save_interval
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        #The agents DNN to predict the most optimal action
        self.net_type = opt.net_type
        self.net = Policy_DDDQN(self.state_dim, self.action_dim, opt).float() if\
            self.net_type == 'ddd-q-net' else Policy_QN(self.state_dim, self.action_dim, opt) #Chapter 2.6.4.7
        self.net = self.net.to(device=self.device)

        # Chapter 2.6.4.2
        self.epsilon = opt.initial_epsilon
        self.epsilon_decay = opt.epsilon_decay
        self.final_epsilon = opt.final_epsilon
        self.curr_step = 0

        #variables for cache
        self.memory = deque(maxlen=opt.memory_size)
        self.batch_size = opt.batch_size #Chapter 2.6.1.3

        #variables for learning
        self.gamma = opt.gamma #Chapter 2.6.4.2
        # Chapter 2.6.1.3
        self.lr = opt.lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.lr_decay = opt.lr_decay
        self.final_lr_frac = opt.final_lr_frac
        lr_lambda = lambda epoch: max(self.lr_decay ** epoch, self.final_lr_frac)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.loss_fn = torch.nn.SmoothL1Loss() #Chapter 2.6.1.5
        self.burnin = opt.burnin
        self.learn_every = opt.learn_every
        self.sync_every = opt.sync_every
        
        if opt.continue_training:
            self.load(opt.load_path, opt.new_epsilon)

    def take_action(self, state):
        '''
        For a given state, choose an epsilon-greedy action
        '''
        #Explore
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)

        #Exploit
        else:
            #state = state.__array__()
            if self.use_cuda:
                #state = torch.tensor(state).cuda()
                state = state.cuda()
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()


        #decrease epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.final_epsilon, self.epsilon)

        #increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        '''
        Add the experience to memory
        '''
        if self.use_cuda:
            state = state.cuda()
            next_state = next_state.cuda()
            action = torch.LongTensor([action]).cuda()
            reward = torch.DoubleTensor([reward]).cuda()
            done = torch.BoolTensor([done]).cuda()
        else:
            action = torch.LongTensor([action])
            reward = torch.DoubleTensor([reward])
            done = torch.BoolTensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        '''
        Sample experience from memory
        '''
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] #Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='target') if self.net_type == 'd-q-net'\
            else self.net(next_state, model='online') #Chapter 2.6.4.7
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1-done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        if self.net_type == 'ddd-q-net':
            self.net.fc_target.load_state_dict(self.net.fc_online.state_dict())
            self.net.value_target.load_state_dict(self.net.value_online.state_dict())
            self.net.adv_target.load_state_dict(self.net.adv_online.state_dict())
        else:
            self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self, epoch=None):
        '''
        Update online action value (Q-value) function with a batch of experiences
        '''
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
            
        if self.curr_step % self.save_every == 0:
            self.save(epoch)

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        #Sample form memory
        state, next_state, action, reward, done = self.recall()

        #Get TD (Time-Difference) Estimate
        td_est = self.td_estimate(state, action)

        #Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        #Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
    
    def save(self, epoch=None):
        save_path = self.save_dir + '/' + f"controller_net_{int(self.curr_step // self.save_every)}.chkpt" if epoch is None \
            else self.save_dir + '/' + f"controller_net_{epoch}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                epsilon=self.epsilon),
            save_path)
        print(f"ddqn saved to {save_path} at step {self.curr_step}")
        
    def load(self, load_path, new_epsilon=False):
        if not os.path.exists(load_path):
            raise ValueError(f"{load_path} does not exist")
            
        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        epsilon = ckp.get('epsilon')
        state_dict = ckp.get('model')
        
        print(f"Loading model at {load_path} with exploration rate {epsilon}")
        self.net.load_state_dict(state_dict)
        if not new_epsilon: self.epsilon = epsilon
        
class Placement_Agent():
    ''' Chapter 3.4.2.2 '''
    def __init__(self, state_dim, action_dim, opt):
        #torch.backends.cudnn.benchmark = False
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = opt.save_dir
        self.save_every = opt.save_interval
        self.use_cuda = torch.cuda.is_available()
        print('Use GPU for training: ', self.use_cuda)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        #The agents DNN to predict the most optimal action
        self.net = Policy_QN(self.state_dim, self.action_dim, opt).float()
        self.net = self.net.to(device=self.device)

        #variables for take_action
        self.epsilon = opt.initial_epsilon
        self.epsilon_decay = opt.epsilon_decay
        self.final_epsilon = opt.final_epsilon
        self.curr_step = 0

        #variables for cache
        self.memory = deque(maxlen=opt.memory_size)
        self.batch_size = opt.batch_size

        #variables for learning
        self.gamma = opt.gamma
        self.lr = opt.lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.lr_decay = opt.lr_decay
        self.final_lr_frac = opt.final_lr_frac
        lr_lambda = lambda epoch: max(self.lr_decay ** epoch, self.final_lr_frac)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda) #https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
        self.loss_fn = torch.nn.SmoothL1Loss() #default-beta is 1.0
        self.burnin = opt.burnin
        self.learn_every = opt.learn_every
        self.sync_every = opt.sync_every
        
        #continue training if desired or use superviced training in prephase
        if opt.continue_training:
            self.load(opt.load_path, opt.new_epsilon)
        self.learn_from_benchmark = opt.learn_from_benchmark
        self.superviced_data_dir = opt.save_dir_superviced_data
        
    def predict(self, next_states):
        '''
        Predict the values for all next states
        '''
        next_states = torch.stack(next_states)
        
        if self.use_cuda:
            next_states = next_states.cuda() #torch.tensor(next_states).cuda() #next_states.clone().detach()
            
        self.net.eval()
        with torch.no_grad():
            self.prediction = self.net(next_states, model='online')[:, 0]
        self.net.train()

    def take_action(self, state_action_dict):
        '''
        For a given state_action_dict, choose an epsilon-greedy action
        '''
        next_actions, next_states = zip(*state_action_dict.items())
        self.predict(next_states)
        
        #Explore
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(0, len(state_action_dict)) #-1

        #Exploit
        else:
            action_idx = torch.argmax(self.prediction).item()

        #decrease epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.final_epsilon, self.epsilon)

        #increment step
        self.curr_step += 1
        return next_actions[action_idx]

    def cache(self, state, next_state, action, reward, done):
        '''
        Add the experience to memory
        '''
        if self.use_cuda:
            state = state.cuda()
            next_state = next_state.cuda()
            action = torch.LongTensor([action]).cuda()
            reward = torch.FloatTensor([reward]).cuda()
            done = torch.BoolTensor([done]).cuda()
        else:
            action = torch.LongTensor([action])
            reward = torch.FloatTensor([reward])
            done = torch.BoolTensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        '''
        Sample experience from memory
        '''
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state_batch):
        current_V = self.net(state_batch, model='online')[np.arange(0, self.batch_size), 0] #V(s)
        return current_V
    
    @torch.no_grad()
    def td_target(self, reward_batch, next_state_batch, done_batch):
        next_V = self.net(next_state_batch, model='target')[np.arange(0, self.batch_size), 0] #V(s')
        return (reward_batch + (1-done_batch.float()) * self.gamma * next_V).float()

    def update_network(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self, epoch=None):
        '''
        Update state value (v-value) function with a batch of experiences
        '''
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
            
        if self.curr_step % self.save_every == 0 and not self.learn_from_benchmark:
            self.save(epoch)

        if self.curr_step < self.burnin and not self.learn_from_benchmark:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        #Sample form memory
        state, next_state, action, reward, done = self.recall()

        #Get TD (Time-Difference) Estimate
        td_est = self.td_estimate(state)

        #Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        #Backpropagate loss through V(s)
        loss = self.update_network(td_est, td_tgt)
        
        return (td_est.mean().item(), loss)
    
    def save(self, epoch=None):
        save_path = self.save_dir + '/' + f"placement_net_{int(self.curr_step // self.save_every)}.chkpt" if epoch is None \
            else self.save_dir + '/' + f"placement_net_{epoch}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                epsilon=self.epsilon),
            save_path)
        print(f"model saved to {save_path} at step {self.curr_step}")
        
    def load(self, load_path, new_epsilon=False):
        if not os.path.exists(load_path):
            raise ValueError(f"{load_path} does not exist")
            
        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        epsilon = ckp.get('epsilon')
        state_dict = ckp.get('model')
        
        print(f"Loading model at {load_path} with exploration rate {epsilon}")
        self.net.load_state_dict(state_dict)
        if not new_epsilon: self.epsilon = epsilon
        
    def load_memory(self, file_name):
        file_path = self.superviced_data_dir + file_name
        print(f"loading superviced data from {file_path}")
        with open(file_path, 'rb') as f:
            self.memory = pickle.load(f)
        
class Benchmark_Agent():
    ''' Chapter 3.4.2.5 '''
    def __init__(self, opt):
        self.memory = deque(maxlen=opt.memory_size)
        self.save_every = opt.save_interval
        self.use_cuda = torch.cuda.is_available()
        self.curr_step = 0
        self.create_supervised_data = opt.create_supervised_data
        self.save_dir = opt.save_dir_superviced_data
    
    def take_action(self, reward_action_dict):
        '''
        Take action with max reward
        '''
        next_actions, next_rewards = zip(*reward_action_dict.items())
        max_reward = max(next_rewards)
        max_reward_idx = next_rewards.index(max_reward)
        self.curr_step += 1
        
        return next_actions[max_reward_idx]

    def cache(self, state, next_state, action, reward, done):
        '''
        Add the experience to memory
        '''
        if self.use_cuda:
            state = state.cuda()
            next_state = next_state.cuda()
            action = torch.LongTensor([action]).cuda()
            reward = torch.FloatTensor([reward]).cuda()
            done = torch.BoolTensor([done]).cuda()
        else:
            action = torch.LongTensor([action])
            reward = torch.FloatTensor([reward])
            done = torch.BoolTensor([done])

        self.memory.append((state, next_state, action, reward, done,))
      
    def learn(self, epoch=None):
        if self.curr_step % self.save_every == 0:
            print('Benchmark: Save memory at current step: ', self.curr_step,
                  'and epoch: ', epoch, 'statement: ', self.create_supervised_data)
            if self.create_supervised_data: self.save_memory()
        return None, None
    
    def save_memory(self):
        save_path = self.save_dir + 'training_data_benchmark.obj'
        with open(save_path, "wb") as f:
            pickle.dump(self.memory, f)
