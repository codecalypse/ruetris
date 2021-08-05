#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 15:28:22 2021

@author: christian brandstaetter
"""

import argparse
import json
from tensorboardX import SummaryWriter
from torchinfo import summary
from src.agent import Controller_Agent
from src.agent import Placement_Agent
from src.agent import Benchmark_Agent
from src.enviroment import Ruetris

def get_args():
    parser = argparse.ArgumentParser(
        '''Implementation of RL-Learning-Agent playing Ruetris''')
    #enviroment variables:
    parser.add_argument('--table_dia', type=int, default=420,
                        help='Diameter of the Table')
    parser.add_argument('--block_size', type=int, default=20,
                        help='Size of a block')
    parser.add_argument('--state_report', type=str, default='reduced',
                        help='Control of state representation: full_bool, full_float, reduced') 
    parser.add_argument('--action_method', type=str, default='controller',
                        help='Kind of actions agent can take in enviroment')
    parser.add_argument('--rnd_pos', type=bool, default=True, 
                        help='For action_method: controller, Rnd initial pos.')
    parser.add_argument('--rnd_rot', type=bool, default=True,
                        help='For action_method: controller, Rnd initial rot.')
    parser.add_argument('--reward_action_dict', type=bool, default=False,
                        help='responding with rewards instead of states for benchmark agent')
    parser.add_argument('--c1', type=int, default=3, help='c1 of reward func.')
    parser.add_argument('--c2', type=int, default=50, help='c2 of reward func.')
    parser.add_argument('--c3', type=int, default=3, help='c3 of reward func.')
    parser.add_argument('--c4', type=int, default=5, help='c4 of reward func.')
    #agent and network variables
    parser.add_argument('--net_type', type=str, default='ddd-q-net',
                        help='valid types: d-q-net, dd-q-net, ddd-q-net. \
                              only for action_method: controller')
    parser.add_argument('--statescale_l1', type=int, default=60,
                        help='Scale-factor s1 of layer 1: nodes_l1=s1*int(input_dim)')
    parser.add_argument('--statescale_l2', type=int, default=40,
                        help='Scale-factor s2 of layer 2: nodes_l2=s2*int(input_dim)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The number of states per batch')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=1.0,
                        help='lr_decay per optimizer_step')
    parser.add_argument('--final_lr_frac', type=float, default=0.1,
                        help='final (min) fraction of the lr')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--initial_epsilon', type=float, default=1)
    parser.add_argument('--final_epsilon', type=float, default=0.08)
    parser.add_argument('--epsilon_decay', type=float, default=0.999999) #0.99999975
    parser.add_argument('--num_episodes', type=int, default=130000)
    parser.add_argument('--memory_size', type=int, default=100000,
                        help='Number of (s,a,n_s,r,d)-tuples in memory')
    parser.add_argument('--burnin', type=int, default=10000, #10000
                        help='Min. experiences before training')
    parser.add_argument('--learn_every', type=int, default=3, #3
                        help='No. of experiences between updates to Q_target')
    parser.add_argument('--sync_every', type=int, default=10000,
                        help='No. of experiences between Q_target & Q_online sync')
    parser.add_argument('--log_path', type=str, default='tensorboard')
    parser.add_argument('--save_dir', type=str, default='trained_models')
    parser.add_argument('--save_interval', type=int, default=1000000) #500000-for controller; 50000-for placement
    parser.add_argument('--render', type=bool, default=False,
                        help='Render the training')
    #arguments for further training:
    parser.add_argument('--load_path', type=str, default='trained_models/model.chkpt')
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--new_epsilon', type=bool, default=True)
    
    #enviroment variables only for manual mode:
    parser.add_argument('--create_supervised_data', type=bool, default=False,
                        help='flag to create supervised data for training')
    parser.add_argument('--save_dir_superviced_data', type=str, default='supervised_data/')
    parser.add_argument('--load_file_superviced_data', type=str, default='benchmark_FullFloat_placement/training_data_benchmark.obj')
    parser.add_argument('--learn_from_benchmark', type=bool, default=False,
                        help='Placement_Agent can use memory from Benchmark_Agent')
    parser.add_argument('--superviced_training_steps', type=int, default=10000,
                        help='Number of learn-calls')
    #parser.add_argument('--supervised_data_size', type=int, default=1000)
    parser.add_argument('--print_state_2_cli', type=bool, default=False)

    args = parser.parse_args()
    return args

def train(opt):
    writer = SummaryWriter(opt.log_path)
    env = Ruetris(opt)
    
    if not opt.reward_action_dict:
        state_dim = env.state_dim
        action_dim = env.action_dim
        agent = Controller_Agent(state_dim, action_dim, opt) if \
                                opt.action_method=='controller' else \
                                Placement_Agent(state_dim, action_dim, opt)
                                
        #make a nice summary of the network:
        model_stats = summary(agent.net, input_size=(opt.batch_size, state_dim)) #agent.net.online
        summary_str = str(model_stats)
        with open('net_summary.txt', 'w') as f:
            f.write(summary_str)
    
    else:
        agent = Benchmark_Agent(opt)
        
    #fill memory of Placement_Agent with memories from Benchmark_Agent
    if isinstance(agent, Placement_Agent) and opt.learn_from_benchmark:
        agent.load_memory(opt.load_file_superviced_data)
        superviced_steps = opt.superviced_training_steps
        for step in range(superviced_steps):
            q, loss = agent.learn()
            if q is not None: writer.add_scalar('Superviced/q-Mean', q, step) 
            if loss is not None: writer.add_scalar('Superviced/Loss', loss, step)
            writer.add_scalar('Superviced/Learning_Rate', agent.optimizer.param_groups[0]['lr'], step)
        agent.learn_from_benchmark = False
        agent.save(0)
    
    episodes = opt.num_episodes
    
    ###------------------ Mainloop: training the model --------------------###
    for epoch in range(episodes):
        state = env.reset()
        if not isinstance(agent, Controller_Agent):
            state_action_dict = env.get_state_action_dict()
        
        while True:         
            #1. Run agent on the state
            action = agent.take_action(state) if isinstance(agent, Controller_Agent) \
                else agent.take_action(state_action_dict)
            
            #2. Agent performs action
            place_state, next_state_action_dict, next_state, reward, done = \
                                                env.step(action, render=opt.render)
            if place_state is not None: state = place_state #needed for benchmark-superviced learning
            #if isinstance(agent, Placement_Agent): state = state_action_dict[action]
            
            #3. Remember
            agent.cache(state, next_state, action, reward, done)
            
            #4. Learn
            q, loss = agent.learn(epoch)
            
            #5. Update state
            state = next_state
            if not isinstance(agent, Controller_Agent):
                state_action_dict = next_state_action_dict
            
            #6. Check if end of game and log important data
            if done:
                writer.add_scalar('Train/Score', env.score, epoch)
                writer.add_scalar('Train/Tetrominoes', env.parts_on_board, epoch)
                writer.add_scalar('Train/Wasted_Places', env.wasted_places, epoch)
                writer.add_scalar('Train/Holes', env.holes, epoch)
                if not isinstance(agent, Benchmark_Agent):
                    writer.add_scalar('Train/Epsilon', agent.epsilon, epoch)
                    writer.add_scalar('Train/Learning_Rate', agent.optimizer.param_groups[0]['lr'], epoch)
                if q is not None: writer.add_scalar('Train/q-Mean', q, epoch) 
                if loss is not None: writer.add_scalar('Train/Loss', loss, epoch)
                break
            
if __name__ == '__main__':
    opt = get_args()
    # saves arguments to config.txt file
    with open('train_config.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
    train(opt)
    