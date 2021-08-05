#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 15:28:22 2021

@author: christian brandstaetter
"""

import argparse
import cv2
import json
from src.agent import Controller_Agent
from src.agent import Placement_Agent
from src.agent import Benchmark_Agent
from src.enviroment import Ruetris

def get_args():
    parser = argparse.ArgumentParser(
        '''Implementation of AI-agent to play Ruetris''')
    parser.add_argument('--fps', type=int, default=24, help='frames per second')
    parser.add_argument('--load_path', type=str, default='trained_models/controller_net_127532.chkpt')
    parser.add_argument('--output', type=str, default='model_performance.avi')
    parser.add_argument('--num_episodes', type=int, default=10)
      
    args = parser.parse_args()
    return args

def test(train_opt, test_opt):
    env = Ruetris(train_opt)
    
    out = cv2.VideoWriter(test_opt.output, cv2.VideoWriter_fourcc(*"XVID"),
                          test_opt.fps, (690, 460))
    
    if not train_opt.reward_action_dict:
        state_dim = env.state_dim
        action_dim = env.action_dim
        agent = Controller_Agent(state_dim, action_dim, train_opt) if \
                                train_opt.action_method=='controller' else \
                                Placement_Agent(state_dim, action_dim, train_opt)
        agent.load(test_opt.load_path)
    
    else:
        agent = Benchmark_Agent()
    
    episodes = test_opt.num_episodes
    
    ###------------------ Mainloop training the model ---------------------###
    for epoch in range(episodes):
        state = env.reset()
        if not isinstance(agent, Controller_Agent):
            state_action_dict = env.get_state_action_dict()
        
        while True:         
            #1. Run agent on the state
            action = agent.take_action(state) if isinstance(agent, Controller_Agent) \
                else agent.take_action(state_action_dict)
            
            #2. Agent performs action (and we screencast it)
            place_state, next_state_action_dict, next_state, reward, done = \
                                                env.step(action, render=True, video=out)
            
            #3. Update state
            state = next_state
            if not isinstance(agent, Controller_Agent):
                state_action_dict = next_state_action_dict
            
            #4. Check if end of game
            if done:
                break
    out.release()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    train_opt = parser.parse_args()
    with open('train_config.txt', 'r') as f:
        train_opt.__dict__ = json.load(f)
    test_opt = get_args()
    test(train_opt, test_opt)
    