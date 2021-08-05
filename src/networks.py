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
    def __init__(self, input_dim, n_actions, opt):
        super(Policy_DDDQN, self).__init__()
        s1 = opt.statescale_l1
        s2 = opt.statescale_l2

        #definition online network
        self.fc_online = nn.Sequential(nn.Linear(input_dim, int(input_dim*s1)), nn.ReLU(),
                                 nn.Linear(int(input_dim*s1), int(input_dim*s2)), nn.ReLU())
        self.value_online = nn.Sequential(nn.Linear(int(input_dim*s2), 1))
        self.adv_online = nn.Sequential(nn.Linear(int(input_dim*s2), n_actions))

        #self._create_weights()
        
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
        ''' Chapter 2.6.1.4 '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform(m.weight, nonlinearity='relu') #He initialization
                #nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, model='online'):
        ''' Figure 2.42 '''
        if model == 'online':
            x = self.fc_online(state)
            v = self.value_online(x)
            adv = self.adv_online(x)
            adv_avg = torch.mean(adv, dim=1, keepdims=True)
            q = v + adv - adv_avg

        elif model == 'target':
            x = self.fc_target(state)
            v = self.value_target(x)
            adv = self.adv_target(x)
            adv_avg = torch.mean(adv, dim=1, keepdims=True)
            q = v + adv - adv_avg
            
        return q

class Policy_QN(nn.Module):
    def __init__(self, input_dim, output_dim, opt):
        #super(Policy_DDQN, self).__init__()
        super().__init__()
        s1 = opt.statescale_l1
        s2 = opt.statescale_l2

        self.online = nn.Sequential(nn.Linear(input_dim, int(input_dim*s1)), nn.ReLU(),
                                    nn.Linear(int(input_dim*s1), int(input_dim*s2)), nn.ReLU(),
                                    nn.Linear(int(input_dim*s2), output_dim))

        self._create_weights()
        self.target = copy.deepcopy(self.online)
        
        #freeze parameters of self.target
        for p in self.target.parameters():
            p.requires_grad = False
            
    def _create_weights(self):
        '''https://pytorch.org/docs/stable/nn.init.html ''' 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.kaiming_uniform_(m.weight,mode='fan_in', nonlinearity='relu') #He initialization
                nn.init.kaiming_normal_(m.weight,mode='fan_in', nonlinearity='relu')
                #nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, model='online'):
        if model == 'online':
            return self.online(state)
        elif model == 'target':
            return self.target(state)
        
class Policy_VN(nn.Module):
    ''' Deprecated and not longer used '''
    def __init__(self, input_dim, output_dim, opt):
        super(Policy_VN, self).__init__()
        s1 = opt.statescale_l1
        s2 = opt.statescale_l2

        self.value = nn.Sequential(nn.Linear(input_dim, int(input_dim*s1)), nn.ReLU(), #nn.ReLU(inplace=True) option for slightly decreased memory usage
                                   nn.Linear(int(input_dim*s1), int(input_dim*s2)), nn.ReLU(),
                                   nn.Linear(int(input_dim*s2), output_dim))
        # self.value = nn.Sequential(nn.Linear(input_dim, 3), nn.Identity(),
        #                            nn.Linear(3, output_dim))

        #self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform(m.weight, nonlinearity='relu') #He initialization
                #nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        return self.value(state)
    