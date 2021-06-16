#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:45:12 2021

@author: christian brandstaetter
"""
import torch.nn as nn
import torch
import copy

class Policy_DDDQN(nn.Module):
    '''
    Dueling-Double-Deep-Q-Network
    '''
    def __init__(self, input_dim, n_actions):
        super(Policy_DDDQN, self).__init__()

        #definition online network
        self.fc_online = nn.Sequential(nn.Linear(input_dim, 126), nn.ReLU(),
                                 nn.Linear(126, 126), nn.ReLU())
        self.value_online = nn.Sequential(nn.Linear(126, 32), nn.ReLU(),
                                          nn.Linear(32, 1))
        self.adv_online = nn.Sequential(nn.Linear(126, 32), nn.ReLU(),
                                        nn.Linear(32, n_actions))

        self._create_weights()
        
        #creating target network
        self.fc_target = copy.deepcopy(self.fc_online)
        self.value_target = copy.deepcopy(self.value_online)
        self.adv_target = copy.deepcopy(self.adv_online)
        
        #freeze parameters of target network
        for p in self.fc_target.parameters():
            p.requires_grad = False
        for p in self.value_target.parameters():
            p.requires_grad = False
        for p in self.adv_target.parameters():
            p.requires_grad = False

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, model):
        if model == 'online':
            x = self.fc_online(state)
            v = self.value_online(x)
            adv = self.adv_online(x)
            adv_avg = torch.mean(adv, dim=1, keepdims=True)
            q = v + adv - adv_avg

        if model == 'target':
            x = self.fc_target(state)
            v = self.value_target(x)
            adv = self.adv_target(x)
            adv_avg = torch.mean(adv, dim=1, keepdims=True)
            q = v + adv - adv_avg
            
        return q

class Policy_QN(nn.Module):
    def __init__(self, input_dim, output_dim):
        #super(Policy_DDQN, self).__init__()
        super().__init__()   
        
        self.online = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(),
                                    nn.Linear(512, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU(),
                                    nn.Linear(128, output_dim))
        
        self._create_weights()
        self.target = copy.deepcopy(self.online)
        
        #freeze parameters of self.target
        for p in self.target.parameters():
            p.requires_grad = False
            
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, model):
        if model == 'online':
            return self.online(state)
        elif model == 'target':
            return self.target(state)
        
class Policy_VN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy_VN, self).__init__()

        self.value = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), #nn.ReLU(inplace=True) option for slightly decreased memory usage
                                   nn.Linear(512, 128), nn.ReLU(),
                                   nn.Linear(128, output_dim)) 

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        return self.value(state)
    