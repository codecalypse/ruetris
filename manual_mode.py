#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 13:45:41 2021

@author: nidhoeggr
"""
import sys
import argparse
from src.enviroment import Ruetris
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Double-Deep-Q-Network to play Ruetris""")
    parser.add_argument('--table_dia', type=int, default=420,
                        help='Diameter of the Table')
    parser.add_argument('--block_size', type=int, default=20,
                        help='Size of a block')
    parser.add_argument('--state_report', type=str, default='full_float',
                        help='Control of state representation: full_bool, full_float, reduced')
    parser.add_argument('--action_method', type=str, default='controller',
                        help='Kind of actions agent can take in enviroment')
    parser.add_argument('--reward_action_dict', type=bool, default=False,
                        help='responding with rewards instead of states for benchmark agent')
    parser.add_argument('--rnd_pos', type=bool, default=True, 
                        help='For action_method: controller, Rnd initial pos.')
    parser.add_argument('--rnd_rot', type=bool, default=True,
                        help='For action_method: controller, Rnd initial rot.')
    parser.add_argument('--c1', type=int, default=2, help='c1 of reward func.')
    parser.add_argument('--c2', type=int, default=50, help='c2 of reward func.')
    parser.add_argument('--c3', type=int, default=3, help='c3 of reward func.')
    parser.add_argument('--c4', type=int, default=5, help='c4 of reward func.')
    parser.add_argument('--create_supervised_data', type=bool, default=True,
                        help='flag to create supervised data for training')
    #parser.add_argument('--supervised_data_size', type=int, default=1000)
    parser.add_argument('--print_state_2_cli', type=bool, default=True)

    args = parser.parse_args()
    return args

def play(opt):
    #print(sys.maxsize)
    np.set_printoptions(threshold=sys.maxsize, linewidth=300)
    game = Ruetris(opt)
    game.render()
    game.render()
    game.manual_mode()

if __name__ == "__main__":
    opt = get_args()
    play(opt)

#torch.set_printoptions(threshold=1000)
