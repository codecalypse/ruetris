#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 13:21:28 2021

@author: christian brandstaetter
"""

import keyboard
import time
import pickle
import torch
import random
import cv2
import numpy as np
import copy as scp
from PIL import Image
from matplotlib import style
from dataclasses import dataclass
from operator import add
from collections import deque

style.use("ggplot")

@dataclass
class filter_options:
    ''' Chapter 3.4.5 '''
    circle_diameter : float = 250
    phis : np.array = np.array([2*np.pi/3-np.pi/2, 4*np.pi/3-np.pi/2, 3*np.pi/2])
    diameters : np.array = np.array([80, 80])

@dataclass
class slot:
    position : np.array = np.array([0, 0])
    index : tuple = (0,0)
    is_used: bool = False

class Ruetris:
    #colors for visualisation with cv
    piece_colors = [
        (254, 254, 254),    #white for free space
        (0, 0, 0),          #black for deadzones
        (255, 255, 0),      #color one for block one
        (147, 88, 254),     #same for other pieces
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)
    ]

    #array with geometry of pieces as numpy arrays
    ''' Chapter 3.2 '''
    pieces = [
        np.array([[2, 2],          #Smashboy
                  [2, 2]]),

        np.array([[0, 3, 0],          #Teewee
                  [3, 3, 3]]),

        np.array([[0, 4, 4],          #Rhode Island Z
                  [4, 4, 0]]),

        np.array([[5, 5, 0],          #Cleveland Z
                  [0, 5, 5]]),

        np.array([[6, 6, 6, 6]]),          #Hero

        np.array([[0, 0, 7],          #Orange Ricky
                  [7, 7, 7]]),

        np.array([[8, 0, 0],          #Blue Ricky
                  [8, 8, 8]])
    ]

    #constructor
    def __init__(self, opt):
        self.table_dia = opt.table_dia
        self.block_size = opt.block_size
        self.state_report = opt.state_report
        self.filter_opt = filter_options()
        self.action_input = opt.action_method
        self.rnd_pos = opt.rnd_pos
        self.rnd_rot = opt.rnd_rot
        self.reward_action_dict = opt.reward_action_dict
        self.c1 = opt.c1
        self.c2 = opt.c2
        self.c3 = opt.c3
        self.c4 = opt.c4
        self.create_supervised_data = opt.create_supervised_data
        #self.supervised_data_size = opt.supervised_data_size
        self.cli_print = opt.print_state_2_cli
        self.create_slot_list()
        self.num_of_cells = int(np.sqrt(len(self.slot_list)))
        self.centre_idx = (self.num_of_cells//2, self.num_of_cells//2)
        self.update_slot_indizes() #now that we know matrix dimension
        self.deadzoneFilter()

        #extra_board for visualization of numbers
        self.info_frame = np.ones((self.num_of_cells * self.block_size,
                    self.num_of_cells * int(self.block_size / 2), 3),
                    dtype=np.uint8) * np.array([62, 11, 78], dtype=np.uint8)
        self.text_color = (255, 255, 255)
        self.reset()
        ''' Kind of observation: Chapter 3.4.2.3 '''
        if self.state_report=='reduced':
            self.state_dim = 4**2+4
        elif self.state_report=='full_bool':
            self.state_dim = self.num_of_cells**2*2
        elif self.state_report=='full_float':
            self.state_dim = self.num_of_cells**2
        ''' Kind of interactin: Chapter 3.4.2.2 '''
        self.action_dim = 9 if self.action_input=='controller' else 1

    def create_slot_list(self):
        self.slot_list = []
        Finished = False
        layer_num = 1
        k = 1
        idx = 0
        self.slot_list.append(slot()) #centre

        right = np.array([self.block_size, 0])
        up = np.array([0, self.block_size])
        left = np.array([-self.block_size, 0])
        down = np.array([0, -self.block_size])

        while not Finished:
            for i in range(layer_num*8):
                if i == 0:
                    pos = self.slot_list[idx].position + right
                elif i // (k+1) == 0:
                    pos = self.slot_list[idx].position + up
                elif i // (k+1) == 1:
                    pos = self.slot_list[idx].position + left
                elif i // (k+1) == 2:
                    pos = self.slot_list[idx].position + down
                elif i // (k+1) == 3:
                    pos = self.slot_list[idx].position + right

                self.slot_list.append(slot(position=pos))
                idx += 1

            if np.linalg.norm(self.slot_list[idx].position)/np.sqrt(2) >= self.table_dia/2:
                Finished = True
            else:
                k += 2
                layer_num += 1

    def update_slot_indizes(self):
        Finished = False
        layer_num = 1
        k = 1
        idx = 0
        self.slot_list[idx].index = self.centre_idx

        right = (0, 1)
        up = (-1, 0)
        left = (0, -1)
        down = (1, 0)

        while not Finished:
            for i in range(layer_num*8):
                if i == 0:
                    idx_tuple = tuple(map(add, self.slot_list[idx].index, right))
                elif i // (k+1) == 0:
                    idx_tuple = tuple(map(add, self.slot_list[idx].index, up))
                elif i // (k+1) == 1:
                    idx_tuple = tuple(map(add, self.slot_list[idx].index, left))
                elif i // (k+1) == 2:
                    idx_tuple = tuple(map(add, self.slot_list[idx].index, down))
                elif i // (k+1) == 3:
                    idx_tuple = tuple(map(add, self.slot_list[idx].index, right))

                idx += 1
                self.slot_list[idx].index = idx_tuple


            if idx == len(self.slot_list)-1:
                Finished = True
            else:
                k += 2
                layer_num += 1

    def create_playboard(self):
        ''' Figure 3.19 '''
        self.playboard = np.zeros((self.num_of_cells, self.num_of_cells),
                                  dtype=float) #empty out playboard

        for slot in self.slot_list:
            self.playboard[slot.index] = slot.is_used

    def create_pieceboard(self, piece, piece_pos):
        ''' Figure 3.20 '''
        pieceboard = np.zeros((self.num_of_cells,
                               self.num_of_cells), dtype=float)
        for i,row in enumerate(piece):
            for j,entry in enumerate(row):
                pieceboard[i + piece_pos["i"]][j + piece_pos["j"]] =\
                    1.0 if bool(entry) else 0.0
        return pieceboard

    def reset(self):
        self.create_playboard()
        self.piece, self.piece_pos = self.new_piece()
        self.score = 0
        self.wasted_places = 0
        self.holes = self.get_holes(self.playboard)
        self.parts_on_board = 0
        self.gameover = False
        return self.get_state(self.playboard, self.piece, self.piece_pos)

    def get_state(self, playboard, piece, piece_pos, cli_print=False):
        ''' Kind of observation: Chapter 3.4.2.3 '''
        if self.state_report == 'full_bool':
            pieceboard = self.create_pieceboard(piece, piece_pos)
            flat_state = np.append(playboard.flatten(), pieceboard.flatten(),
                                   axis=0)
            if cli_print:
                print(playboard)
                print(pieceboard)
            return torch.FloatTensor(flat_state)
        
        elif self.state_report == 'full_float':
            if cli_print:
                print(self.get_full_float_state(playboard, piece, piece_pos)) 
                print('\n')
            return torch.FloatTensor(self.get_full_float_state(\
                                    playboard, piece, piece_pos).flatten())

        elif self.state_report == 'reduced':
            distances_vec = self.calc_distances(playboard, piece, piece_pos)
            if cli_print:
                print(distances_vec)
            return torch.FloatTensor(np.array(distances_vec).flatten())
        
    def get_state_no_piece(self, playboard):
        ''' Kind of observation, but with no piece: Chapter 3.4.2.3 '''
        if self.state_report == 'full_bool':
            pieceboard = np.zeros(playboard.shape) #no information about piece
            flat_state = np.append(playboard.flatten(), pieceboard.flatten(),
                                   axis=0)
            return torch.FloatTensor(flat_state)
        
        elif self.state_report == 'full_float':
            return torch.FloatTensor(playboard.flatten())

        elif self.state_report == 'reduced':
            flat_state = np.zeros(20) #and no information for reduced
            return torch.FloatTensor(flat_state)
    
    def get_full_float_state(self, playboard, piece, piece_pos):
        ''' Figure 3.21 '''
        float_board = scp.copy(playboard)
        for i,row in enumerate(piece):
            for j,entry in enumerate(row):
                condition_A = bool(playboard[i + piece_pos["i"]][j + piece_pos["j"]])
                condition_B = bool(entry)
                if condition_B and condition_A:
                    float_board[i + piece_pos["i"]][j + piece_pos["j"]] = -1.0
                elif condition_B and not condition_A:
                    float_board[i + piece_pos["i"]][j + piece_pos["j"]] = 0.5
        return float_board
    
    def get_state_action_dict(self):
        ''' All possible storage positions: Figure 3.24 '''
        state_action_dict = {}
        piece = scp.copy(self.piece)
        if self.piece_idx == 0: #smashboy
            num_rotations = 1
        elif self.piece_idx == 2 or self.piece_idx == 3 or self.piece_idx == 4:
            num_rotations = 2
        else:
            num_rotations = 4
        
        for rot in range(num_rotations):
            valid_i = self.num_of_cells - len(piece)
            valid_j = self.num_of_cells - len(piece[0])
            for x in range((valid_i+1)*(valid_j+1)):
                i = x // (valid_j+1)
                j = x % (valid_j+1)
                pos = {'i': i, 'j': j}
                out_of_boundary, overlapping = self.check_action(pos, piece)
                if not out_of_boundary and not overlapping:
                    state_action_dict[(i, j, rot)] = \
                        self.get_state(self.playboard, piece, pos) if not \
                        self.reward_action_dict else self.action_place_dummy(self.playboard, piece, pos)
            _, piece = self.rot_action('right', piece)
        return state_action_dict
    
    def action_place_dummy(self, board, piece, pos):
        ''' Calculation of rewards without changing state of environment.
            This is for the 'Benchmark-Agent' Chapter 3.4.2.5 '''
        score_inc = 0
        next_board = scp.deepcopy(board)
        
        for i,row in enumerate(piece):
            for j,entry in enumerate(row):
                if bool(entry):
                    index = (i+pos['i'], j+pos['j'])
                    next_board[index] = 1

        distances_vec = self.calc_distances(board, piece, pos)
        n_neighbours = distances_vec[0:16].count(0)
        x_i_norm = sum(distances_vec[16:])/4

        current_wasted_places = self.calc_wasted_places(next_board)
        n_wasted_places = current_wasted_places - self.wasted_places

        score_inc += self.eval_measurements(x_i_norm, n_wasted_places,
                                            n_neighbours)

        return score_inc

    def calc_distances(self, playboard, piece, piece_pos):
        ''' Chapter 3.4.2.1 and Figure 3.12 '''
        distances_vec = []
        base_vec = []
        for i,row in enumerate(piece):
            for j,entry in enumerate(row):
                if bool(entry):
                    distances_vec.append(self.count4directions(i,j,
                                                playboard, piece_pos))
                    base_vec.append(self.distance2base(i,j, piece_pos))
        flat_distances = [distance for sublist in distances_vec for distance in sublist]
        return flat_distances + base_vec

    def count4directions(self, i, j, playboard, piece_pos):
        ''' Chapter 3.4.2.3, Figure 3.22 and 3.23 '''
        start_idx = (i + piece_pos['i'], j + piece_pos['j'])
        directions = [(1,0), (0,1), (-1,0), (0,-1)] if playboard[start_idx] \
                        else [(-1,0), (0,-1), (1,0), (0,1)]
        looking_for = not playboard[start_idx]
        distances = []
        for direction in directions:
            distance = 1 if playboard[start_idx] else 0
            idx_tuple = tuple(map(add, start_idx, direction))
            while True:
                is_i_min = not idx_tuple[0]>=0
                is_j_min = not idx_tuple[1]>=0
                is_i_max = not idx_tuple[0]<self.num_of_cells
                is_j_max = not idx_tuple[1]<self.num_of_cells
                condition = is_i_min or is_j_min or is_i_max or is_j_max
                if condition or playboard[idx_tuple]==looking_for:
                    c = self.num_of_cells
                    distances.append(-distance/c if playboard[start_idx] \
                                     else distance/c)
                    break
                else:
                    distance += 1
                    idx_tuple = tuple(map(add, idx_tuple, direction))
        return distances

    def distance2base(self, i, j, piece_pos): #manhatten-distance or squared-distance would speed up things
        ''' Equation 3.1 '''
        target_idx = (i + piece_pos['i'], j + piece_pos['j'])
        for slot in self.slot_list:
            if slot.index == target_idx:
                return np.linalg.norm(slot.position)/self.table_dia/2

    def new_piece(self, piece_idx=None):
        ''' Chapter 3.4.2.2, struct 'transition-possibility 
            vector' (Eqation 3.4) not implemented '''
        if piece_idx is None:
            self.piece_idx = random.randrange(0,len(self.pieces))
        else:
            self.piece_idx = piece_idx

        piece = self.pieces[self.piece_idx]
        
        if self.rnd_rot and self.piece_idx != 0:
            num_rot = 2 if self.piece_idx == 2 or self.piece_idx == 3 or \
                self.piece_idx == 4 else 4
            for _ in range(random.randrange(0, num_rot)):
                piece = piece.transpose()
                piece = np.fliplr(piece)
        
        if not self.rnd_pos:
            piece_pos = {'i': 1, 'j': 1}
        else:
            valid_i = self.num_of_cells - len(piece)
            valid_j = self.num_of_cells - len(piece[0])
            piece_pos = {'i': random.randrange(1, valid_i),
                         'j': random.randrange(1, valid_j)}
        return piece, piece_pos

    def deadzoneFilter(self):
        ''' First mentioned in chapter 3.4.2.1 '''
        self.boundary_filter() #make table-shape
        self.inner_filter()  #filter for inner-deadzone
        self.stand_filter() #filter for distributed deadzones

    def boundary_filter(self):
        for slot in self.slot_list:
            pos = slot.position
            if np.linalg.norm(pos) >= self.table_dia/2:
                slot.is_used = True

    def inner_filter(self):
        inner_dia = self.filter_opt.diameters[0]
        for slot in self.slot_list:
            pos = slot.position
            if np.linalg.norm(pos) <= inner_dia/2:
                slot.is_used = True

    def stand_filter(self):
        dia = self.filter_opt.diameters[1]
        circle_dia = self.filter_opt.circle_diameter
        phis = self.filter_opt.phis

        for slot in self.slot_list:
            pos = slot.position
            for phi in phis:
                zone_pos = np.array([circle_dia/2*np.cos(phi),
                                     circle_dia/2*np.sin(phi)])
                rel_vec = pos - zone_pos
                if np.linalg.norm(rel_vec) <= dia/2:
                    slot.is_used = True

    def manual_mode(self):
        ''' For the manual_mode.py-script, Chapter 4.4.3 '''
        self.supervised_data = deque() #maxlen=self.supervised_data_size
        choosen_action = -1
        score_inc = 0
        last_state = self.get_state(self.playboard, self.piece,
                                    self.piece_pos, self.cli_print)
        
        while True:
            key_was_hit = False
            if keyboard.is_pressed('w'):
                score_inc, position = self.move_action('up')
                self.score += score_inc
                self.piece_pos = position
                self.render()
                key_was_hit = True
                choosen_action = 0
                time.sleep(0.2)
            if keyboard.is_pressed('d'):
                score_inc, position = self.move_action('right')
                self.score += score_inc
                self.piece_pos = position
                self.render()
                key_was_hit = True
                choosen_action = 3
                time.sleep(0.2)
            if keyboard.is_pressed('s'):
                score_inc, position = self.move_action('down')
                self.score += score_inc
                self.piece_pos = position
                self.render()
                key_was_hit = True
                choosen_action = 1
                time.sleep(0.2)
            if keyboard.is_pressed('a'):
                score_inc, position = self.move_action('left')
                self.score += score_inc
                self.piece_pos = position
                self.render()
                key_was_hit = True
                choosen_action = 2
                time.sleep(0.2)
            if keyboard.is_pressed('e'):
                score_inc, self.piece = self.rot_action('left')
                self.score += score_inc
                self.render()
                key_was_hit = True
                choosen_action = 5
                time.sleep(0.2)
            if keyboard.is_pressed('q'):
                score_inc, self.piece = self.rot_action('right')
                self.score += score_inc
                self.render()
                key_was_hit = True
                choosen_action = 4
                time.sleep(0.2)
            if keyboard.is_pressed('r'):
                score_inc, self.playboard = self.action_rot_table_left()
                self.score += score_inc
                self.render()
                key_was_hit = True
                choosen_action = 7
                time.sleep(0.2)
            if keyboard.is_pressed('t'):
                score_inc, self.playboard = self.action_rot_table_right()
                self.score += score_inc
                self.render()
                key_was_hit = True
                choosen_action = 6
                time.sleep(0.2)
            if keyboard.is_pressed('f'):
                score_inc, self.gameover, self.playboard = self.action_place_piece()
                self.score += score_inc
                if not self.gameover:
                    self.piece, self.piece_pos = self.new_piece()
                else:
                    self.reset()
                self.render()
                key_was_hit = True
                choosen_action = 8
                time.sleep(0.2)
            if keyboard.is_pressed('x'):
                with open("supervised_data/training_data.obj", "wb") as f:
                    pickle.dump(self.supervised_data, f)
                break
            
            if key_was_hit:
                next_state = self.get_state(self.playboard, self.piece,
                                            self.piece_pos, self.cli_print)
                self.supervised_data.append((last_state, next_state,
                                             choosen_action, score_inc,
                                             self.gameover,))
                last_state = next_state

    def check_action(self, wanted_position, piece=np.zeros((2,4))):
        ''' Mentioned in chapter 4.4.2 (Tab. 4.3) '''
        if not piece.any():
            piece = self.piece
        overlapping = False
        out_of_boundary = False

        for i,row in enumerate(piece):
            for j,entry in enumerate(row):
                wanted_idx = (i+wanted_position['i'], j+wanted_position['j'])
                is_i_min = not wanted_idx[0]>=0
                is_j_min = not wanted_idx[1]>=0
                is_i_max = not wanted_idx[0]<self.num_of_cells
                is_j_max = not wanted_idx[1]<self.num_of_cells
                condition = is_i_min or is_j_min or is_i_max or is_j_max
                if bool(entry) and condition:
                    out_of_boundary = True
                elif bool(entry) and self.playboard[wanted_idx]:
                    overlapping = True
        return out_of_boundary, overlapping

    def move_action(self, intent):
        ''' Chapter 3.4.2.2, Figure 3.17 and Chapter 4.4.2 '''
        wanted_move = scp.copy(self.piece_pos)
        score_inc = 0
        if intent == 'up':
            wanted_move['i'] -= 1
        elif intent == 'down':
            wanted_move['i'] += 1
        elif intent == 'left':
            wanted_move['j'] -= 1
        elif intent == 'right':
            wanted_move['j'] += 1

        out_of_boundary, overlapping = self.check_action(wanted_move)

        if out_of_boundary:
            wanted_move = self.piece_pos
            score_inc -= 4
        elif overlapping:
            score_inc -= 2
        else:
            score_inc -= 1

        return score_inc, wanted_move

    def rot_action(self, intent, rot_piece=np.zeros((2,4))):
        ''' Chapter 3.4.2.2, Figure 3.17 and Chapter 4.4.2 '''
        if not rot_piece.any():
            rot_piece = scp.copy(self.piece)
        score_inc = 0
        if intent == 'right':
            rot_piece = rot_piece.transpose()
            rot_piece = np.fliplr(rot_piece)
        if intent == 'left':
            for i,_ in enumerate(rot_piece):
                rot_piece[i] = np.flip(rot_piece[i])
            rot_piece = rot_piece.transpose()

        out_of_boundary, overlapping = self.check_action(self.piece_pos,
                                                         rot_piece)

        if out_of_boundary:
            rot_piece = self.piece
            score_inc -= 4
        elif overlapping:
            score_inc -= 2
        else:
            score_inc -= 1

        return score_inc, rot_piece

    def action_rot_table_right(self):
        ''' Chapter 3.4.2.2, Figure 3.17 and Chapter 4.4.2 '''
        score_inc = 0
        score_inc -= 3
        new_playboard = scp.copy(self.playboard)
        new_playboard = new_playboard.transpose()
        new_playboard = np.fliplr(new_playboard)
        return score_inc, new_playboard

    def action_rot_table_left(self):
        ''' Chapter 3.4.2.2, Figure 3.17 and Chapter 4.4.2 '''
        score_inc = 0
        score_inc -= 3
        new_playboard = scp.copy(self.playboard)
        for i,_ in enumerate(new_playboard):
            new_playboard[i] = np.flip(new_playboard[i])
        new_playboard = new_playboard.transpose()
        return score_inc, new_playboard

    def action_place_piece(self):
        ''' Chapter 3.4.2.2, Figure 3.17 and Chapter 4.4.2 '''
        gameover = True
        score_inc = 0
        out_of_boundary, overlapping = self.check_action(self.piece_pos)
        board = scp.deepcopy(self.playboard)

        if out_of_boundary:
            raise Exception('line443: out_of_boundary should never happen')
        elif overlapping:
            score_inc -= 500
        else:
            distances_vec = self.calc_distances(board, self.piece,
                                                self.piece_pos)
            n_neighbours = distances_vec[0:16].count(0)
            x_i_norm = sum(distances_vec[16:])/4

            for i,row in enumerate(self.piece):
                for j,entry in enumerate(row):
                    if bool(entry):
                       index = (i+self.piece_pos['i'], j+self.piece_pos['j'])
                       board[index] = 1

            self.parts_on_board += 1
            current_wasted_places = self.calc_wasted_places(board) #if self.c2 != 0 else 0 ... would speed up things
            n_wasted_places = current_wasted_places - self.wasted_places
            self.wasted_places = current_wasted_places
            self.holes = self.get_holes(board)
            win_cond = self.holes < 4
            if win_cond:
                score_inc += 500
            else:
                gameover = False
            score_inc += self.eval_measurements(x_i_norm, n_wasted_places,
                                                n_neighbours)

        return score_inc, gameover, board

    def eval_measurements(self, x_i_norm, n_wasted_places, n_neighbours):
        ''' Equation 3.2 '''
        reward = round(1/x_i_norm*self.c1) - n_wasted_places*self.c2 + \
                 n_neighbours*self.c3 + self.c4
        # print('c1: ', round(1/x_i_norm*c1))
        # print('c2: ', n_wasted_places)
        # print('c3: ', n_neighbours)
        # print('comu: ', reward)
        return reward

    def choose_action(self, action_num):
        ''' Interface for Controller-Agent'''
        playboard = self.playboard
        piece = self.piece
        pos = self.piece_pos
        score_inc = 0
        gameover = False
        if action_num == 0:
            score_inc, pos = self.move_action('up')
        elif action_num == 1:
            score_inc, pos = self.move_action('down')
        elif action_num == 2:
            score_inc, pos = self.move_action('left')
        elif action_num == 3:
            score_inc, pos = self.move_action('right')
        elif action_num == 4:
            score_inc, piece = self.rot_action('right')
        elif action_num == 5:
            score_inc, piece = self.rot_action('left')
        elif action_num == 6:
            score_inc, playboard = self.action_rot_table_right()
        elif action_num == 7:
            score_inc, playboard = self.action_rot_table_left()
        elif action_num == 8:
            score_inc, gameover, playboard = self.action_place_piece()
            piece, pos = self.new_piece()
        else:
            raise Exception('action_num not valid')

        return playboard, piece, pos, score_inc, gameover

    def get_holes(self, board):
        '''
        Calculation of present holes on the board
        '''
        num_holes = 0
        for i,row in enumerate(board):
            for j,entry in enumerate(row):
                if not bool(entry):
                    num_holes += 1
        return num_holes

    def calc_wasted_places(self, board): #speed up things with a smaller view near piece
        ''' Chapter 3.4.2.1, Figure 3.12 and 3.13 '''
        wasted_place = 0
        mask = board != 0
        xDim, yDim = board.shape
        for i in range(xDim):
            for j in range(yDim):
                if mask[i][j]: #continue if it is a taken place, else see if wasted
                    continue
                #neighbours in -x, x, -y, y direction for waste_piece
                try: #catch indexes outside of dim with try-except-routines
                    neighbours0 = [mask[i-1][j], mask[i+1][j], mask[i][j-1],
                                   mask[i][j+1]]
                    #- 1 -
                    #1 x 1
                    #- 1 -
                except:
                    neighbours0 = [False]
                try:
                    neighbours1 = [mask[i-1][j], mask[i+2][j], mask[i][j-1],
                                   mask[i][j+1], mask[i+1][j-1], mask[i+1][j+1],
                                   not mask[i+1][j]]
                    #- 1 -
                    #1 x 1
                    #1 0 1
                    #- 1 -
                except:
                    neighbours1 = [False]
                try:
                    neighbours2 = [mask[i-1][j], mask[i+1][j], mask[i][j-1],
                                   mask[i][j+2], mask[i-1][j+1], mask[i+1][j+1],
                                   not mask[i][j+1]]
                    #- 1 1 -
                    #1 x 0 1
                    #- 1 1 -
                except:
                    neighbours2 = [False]
                try:
                    neighbours3 = [mask[i-1][j], mask[i+3][j], mask[i][j-1],
                                   mask[i][j+1], mask[i+1][j-1], mask[i+1][j+1],
                                   mask[i+2][j-1], mask[i+2][j+1],
                                   not mask[i+1][j], not mask[i+2][j]]
                    #- 1 -
                    #1 x 1
                    #1 0 1
                    #1 0 1
                    #- 1 -
                except:
                    neighbours3 = [False]
                try:
                    neighbours4 = [mask[i-1][j], mask[i+1][j], mask[i][j-1],
                                   mask[i][j+3], mask[i-1][j+1], mask[i+1][j+1],
                                   mask[i-1][j+2], mask[i+1][j+2],
                                   not mask[i][j+1], not mask[i][j+2]]
                    #- 1 1 1 -
                    #1 x 0 0 1
                    #- 1 1 1 -
                except:
                    neighbours4 = [False]
                try:
                    neighbours5 = [mask[i-1][j], mask[i+2][j], mask[i][j-1],
                                   mask[i][j+2], mask[i+1][j-1], mask[i+1][j+1],
                                   mask[i-1][j+1], not mask[i][j+1],
                                   not mask[i+1][j]]
                    #- 1 1 -
                    #1 x 0 1
                    #1 0 1 -
                    #- 1 - -
                except:
                    neighbours5 = [False]
                try:
                    neighbours6 = [mask[i-1][j], mask[i+1][j], mask[i][j-1],
                                   mask[i][j+2], mask[i-1][j+1], mask[i+2][j+1],
                                   mask[i+1][j+2], not mask[i][j+1],
                                   not mask[i+1][j+1]]
                    #- 1 1 -
                    #1 x 0 1
                    #- 1 0 1
                    #- - 1 -
                except:
                    neighbours6 = [False]
                try:
                    neighbours7 = [mask[i-1][j], mask[i+2][j], mask[i][j-1],
                                   mask[i][j+1], mask[i+1][j-1], mask[i+1][j+2],
                                   mask[i+2][j+1], not mask[i+1][j],
                                   not mask[i+1][j+1]]
                    #- 1 - -
                    #1 x 1 -
                    #1 0 0 1
                    #- 1 1 -
                except:
                    neighbours7 = [False]
                try:
                    neighbours8 = [mask[i-1][j], mask[i+1][j], mask[i][j-1],
                                   mask[i][j+2], mask[i+1][j+1], mask[i-2][j+1],
                                   mask[i-1][j+2], not mask[i][j+1],
                                   not mask[i-1][j+1]]
                    #- - 1 -
                    #- 1 0 1
                    #1 x 0 1
                    #- 1 1 -
                except:
                    neighbours8 = [False]
                    
                #if some neighbour are all true: places are taken -> wasted place
                #these masks represent all possible shapes of wasted placed. 
                #this depends on possilbel shapes, here it depends on tetrominoes.
                if np.all(neighbours0):
                    wasted_place += 1
                    mask[i][j] = 1
                elif np.all(neighbours1):
                    wasted_place += 2
                    mask[i][j] = 1
                    mask[i+1][j] = 1
                elif np.all(neighbours2):
                    wasted_place += 2
                    mask[i][j] = 1
                    mask[i][j+1] = 1
                elif np.all(neighbours3):
                    wasted_place += 3
                    mask[i][j] = 1
                    mask[i+1][j] = 1
                    mask[i+2][j] = 1
                elif np.all(neighbours4):
                    wasted_place += 3
                    mask[i][j] = 1
                    mask[i][j+1] = 1
                    mask[i][j+2] = 1
                elif np.all(neighbours5):
                    wasted_place += 3
                    mask[i][j] = 1
                    mask[i][j+1] = 1
                    mask[i+1][j] = 1
                elif np.all(neighbours6):
                    wasted_place += 3
                    mask[i][j] = 1
                    mask[i][j+1] = 1
                    mask[i+1][j+1] = 1
                elif np.all(neighbours7):
                    wasted_place += 3
                    mask[i][j] = 1
                    mask[i+1][j] = 1
                    mask[i+1][j+1] = 1
                elif np.all(neighbours8):
                    wasted_place += 3
                    mask[i][j] = 1
                    mask[i-1][j+1] = 1
                    mask[i][j+1] = 1
        return wasted_place

    def step(self, action, render=True, video=None):
        ''' Interface for all agents with the environment'''
        if self.action_input=='controller':
            self.playboard, self.piece, self.piece_pos, score_inc, self.gameover \
                                    = self.choose_action(action)
            next_state = self.get_state(self.playboard, self.piece, self.piece_pos)
            self.score += score_inc
            next_state_action_dict = None
            place_state = None
        else:
            i, j, num_rotations = action
            for _ in range(num_rotations):
                _, self.piece = self.rot_action('right')
            self.piece_pos = {'i': i, 'j': j}
            place_state = self.get_state(self.playboard, self.piece, self.piece_pos)
            score_inc, gameover, self.playboard = self.action_place_piece()
            self.score += score_inc
            next_state = self.get_state_no_piece(self.playboard) #torch.FloatTensor(self.playboard.flatten())
            self.piece, self.piece_pos = self.new_piece()
            next_state_action_dict = self.get_state_action_dict()
            self.gameover = gameover if len(next_state_action_dict) > 0 else True
            #next_state = self.get_state(self.playboard, self.piece, self.piece_pos) #with new piece
            
        self.render(video) if render else None
        return place_state, next_state_action_dict, next_state, score_inc, self.gameover

    def get_board_visu(self):
        board = scp.copy(self.playboard)
        for i,row in enumerate(self.piece):
            for j,entry in enumerate(row):
                if bool(entry):
                    index = (i+self.piece_pos['i'], j+self.piece_pos['j'])
                    board[index] = entry
        return board

    def render(self, video=None):
        img = [self.piece_colors[int(p)] for row in self.get_board_visu() for p in row]
        cells = self.num_of_cells
        a = self.block_size
        img = np.array(img).reshape((cells, cells, 3)).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        img = img.resize((cells*a, cells*a), Image.NEAREST)
        img = np.array(img)
        #draw boarders
        img[[i*a for i in range(cells)], :, :] = 0
        img[:, [i*a for i in range(cells)], :] = 0

        #append info box
        img = np.concatenate((img, self.info_frame), axis=1)

        #display stuff
        cv2.putText(img, "Score:", (cells*a + int(a/2), a), fontScale=1.0,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, color=self.text_color)
        cv2.putText(img, str(self.score),
                    (cells*a + int(a/2), 2*a), fontScale=1.0,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, color=self.text_color)
        cv2.putText(img, "Pieces:", (cells*a + int(a/2), 4*a), fontScale=1.0,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, color=self.text_color)
        cv2.putText(img, str(self.parts_on_board),
                    (cells*a + int(a/2), 5*a), fontScale=1.0,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, color=self.text_color)
        cv2.putText(img, "Wasted:", (cells*a + int(a/2), 7*a), fontScale=1.0,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, color=self.text_color)
        cv2.putText(img, str(self.wasted_places),
                    (cells*a + int(a/2), 8*a), fontScale=1.0,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, color=self.text_color)
        cv2.putText(img, "Holes:", (cells*a + int(a/2), 10*a), fontScale=1.0,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, color=self.text_color)
        cv2.putText(img, str(self.holes),
                    (cells*a + int(a/2), 11*a), fontScale=1.0,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, color=self.text_color)

        if video is not None:
            #print(img.shape)
            video.write(img)

        cv2.imshow("Deep Q-Learning Ruetris!", img)
        cv2.waitKey(1)
