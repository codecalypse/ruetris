#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 15:28:22 2021

@author: christian brandstaetter
"""

import argparse
import cv2
from src.agent import Controller_Agent
from src.agent import Placement_Agent
from src.agent import Analytical_Agent
from src.enviroment import Ruetris

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Double-Deep-Q-Network to play Ruetris""")
    parser.add_argument('--table_dia', type=int, default=420,
                        help='Diameter of the Table')
    parser.add_argument('--block_size', type=int, default=20,
                        help='Size of a block')
    parser.add_argument('--fps', type=int, default=24, help='frames per second')
    parser.add_argument('--load_path', type=str, default='trained_models/model.chkpt')
    parser.add_argument('--output', type=str, default='model_performance.avi')
    parser.add_argument('--state_report', type=str, default='full',
                        help='Control of state representation') 
    parser.add_argument('--action_method', type=str, default='placement',
                        help='Kind of actions agent can take in enviroment')
    parser.add_argument('--net_type', type=str, default='dd-q-net',
                        help='valid types: d-q-net, dd-q-net, ddd-q-net')
    parser.add_argument('--rnd_pos', type=bool, default=True, 
                        help='For action_method: controller, Rnd initial pos.')
    parser.add_argument('--rnd_rot', type=bool, default=True,
                        help='For action_method: controller, Rnd initial rot.')
    parser.add_argument('--reward_action_dict', type=bool, default=False,
                        help='responding with rewards instead of states for analytical agent')
    parser.add_argument('--num_episodes', type=int, default=10)
    
    #parameters just for training, but also needed here for initialization 
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The number of states per batch')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--initial_epsilon', type=float, default=1)
    parser.add_argument('--final_epsilon', type=float, default=1e-3)
    parser.add_argument('--epsilon_decay', type=float, default=0.999975)
    parser.add_argument('--memory_size', type=int, default=40000,
                        help='Number of (s,a,n_s,r)-tuples in memory')
    parser.add_argument('--burnin', type=int, default=10000,
                        help='Min. experiences before training')
    parser.add_argument('--learn_every', type=int, default=3,
                        help='No. of experiences between updates to Q_target')
    parser.add_argument('--sync_every', type=int, default=5000,
                        help='No. of experiences between Q_target & Q_online sync')
    parser.add_argument('--log_path', type=str, default='tensorboard')
    parser.add_argument('--save_dir', type=str, default='trained_models')
    parser.add_argument('--save_interval', type=int, default=500000)

    args = parser.parse_args()
    return args

def test(opt):
    env = Ruetris(table_dia=opt.table_dia, block_size=opt.block_size,
                  state_report=opt.state_report, action_input=opt.action_method,
                  rnd_pos=opt.rnd_pos, rnd_rot=opt.rnd_rot,
                  reward_action_dict=opt.reward_action_dict)
    
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"XVID"), opt.fps,
                          (690, 460))
    
    if not opt.reward_action_dict:
        state_dim = env.state_dim
        action_dim = env.action_dim
        agent = Controller_Agent(state_dim, action_dim, opt) if \
                                opt.action_method=='controller' else \
                                Placement_Agent(state_dim, action_dim, opt)
        agent.load(opt.load_path)
    
    else:
        agent = Analytical_Agent()
    
    episodes = opt.num_episodes
    
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
            next_state_action_dict, next_state, reward, done = \
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
    opt = get_args()
    test(opt)
    