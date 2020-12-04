#!/usr/bin/python

from ab_cutoff import alpha_beta_cutoff

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random
import copy
# import tensorflow as tf

# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""

    # def __init__(self):
    #
    #     # self.model = tf.keras.models.load_model("trainedModel")
    #
    #     self.move_map = {
    #     0 : "U",
    #     1 : "D",
    #     2 : "L",
    #     3 : "R"
    #     }

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        return alpha_beta_cutoff(asp, 9, eval_func)


        # result = self.model(self.parse(asp))
        #
        # best_move = tf.argmax(result)
        #
        # return self.move_map[result]

    def parse(self, TronP):
        featureNumMap = {
        "#" : 0,
        "1" : 1,
        "2" : 2,
        " " : 3,
        "x" : 4,
        "*" : 5,
        "@" : 6,
        "^" : 7,
        "!" : 8,
        "-" : 9
        }

        max_board_size = 13
        #board = TronP.board

        np_board = np.zeros(shape=(max_board_size, max_board_size), dtype=int)

        for i in range(len(board)):
            for j in range(len(board[0])):
                np_board[i][j] = featureNumMap[board[i][j]]

        np_board = np_board.flatten()
        onehot_board = np.eye(10)[np_board]
        flattened_onehot_board = onehot_board.flatten()

        non_board_state = np.zeros(shape=(11,), dtype=int)

        #Set ptm
        non_board_state[0] = int(TronP.player_to_move())

        #Set player armors
        non_board_state[1] = TronP.player_has_armor(0)
        non_board_state[2] = TronP.player_has_armor(1)

        #Set player boost
        p1_boost = TronP.get_remaining_turns_speed(0)
        if(p1_boost != 0):
            non_board_state[2+p1_boost] = 1

        p2_boost = TronP.get_remaining_turns_speed(1)
        if(p2_boost != 0):
            non_board_state[6+p2_boost] = 1


        return np.concatenate( (flattened_onehot_board, non_board_state), axis=0)
def adjacent_coords(board, loc):
    coords = []
    if loc[0] > 1:
        coords.append((loc[0] - 1, loc[1]))
    if loc[0] < (len(board) - 2):
        coords.append((loc[0] +1 , loc[1]))
    if loc[1] > 1:
        coords.append((loc[0], loc[1] - 1))
    if loc[1] < (len(board[0]) - 2):
        coords.append((loc[0], loc[1] + 1))

    return coords

def find_player(board, player_num):

    val = str(player_num)
    for x in range(len(board)):
        for y in range(len(board[0])):
            if board[x][y] == val:
                return (x,y)
    return None

def eval_func(TronP):

    # store the visited nodes and their distance from start
    #to check easily if already visited
    p1_visited_dict = {}
    p1_curr_loc = find_player(TronP.board, 1)
    p1_frontier = [p1_curr_loc,]
    p1_visited_dict[p1_curr_loc] = 0

    #to check easily if already visited
    p2_visited_dict = {}
    p2_curr_loc = find_player(TronP.board, 2)
    p2_frontier = [p2_curr_loc,]
    p2_visited_dict[p2_curr_loc] = 0

    numAvailSpaces = 0

    for thing in TronP.board:
        for b in thing:
            if b not in ["#", "-", 'x', '1', '2']:
                numAvailSpaces += 1


    #player_one
    while len(p1_frontier) > 0 or len(p2_frontier) > 0:

        p1_next_frontier = []
        p2_next_frontier = []

        while(len(p1_frontier) != 0):
            curr = p1_frontier.pop()
            adj = adjacent_coords(TronP.board, curr)

            for z in adj:
                if TronP.board[z[0]][z[1]] in ["#", "-", "x", "1", "2"]:
                    continue
                if z in p1_visited_dict:
                    continue
                if z in p2_visited_dict and p1_visited_dict[curr] + 1 > p2_visited_dict[z]:
                    continue
                else:
                    p1_next_frontier.append(z)
                    p1_visited_dict[z] = p1_visited_dict[curr] + 1

        p1_frontier = p1_next_frontier

        while( len(p2_frontier)):
            curr = p2_frontier.pop()
            adj = adjacent_coords(TronP.board, curr)
            for z in adj:
                if TronP.board[z[0]][z[1]] in ["#", "-", "x", "1", "2"]:
                    continue
                if z in p2_visited_dict:
                    continue
                if z in p1_visited_dict and p2_visited_dict[curr] + 1 > p1_visited_dict[z]:
                    continue
                else:
                    p2_next_frontier.append(z)
                    p2_visited_dict[z] = p2_visited_dict[curr] + 1
        p2_frontier = p2_next_frontier

    markedBoard = copy.deepcopy(TronP.board)



    p1Set = list(p1_visited_dict.keys())
    p2Set = list(p2_visited_dict.keys())
    # print(p1Set)
    # print(p2Set)
    p1Betters = []
    for thing in p1Set:
        if thing not in p2Set:
            p1Betters.append(thing)
    p2Betters =  []

    for thing in p2Set:
        if thing not in p1Set:
            p2Betters.append(thing)

    # print("betters")
    # print(p1Betters)
    # print(p2Betters)


    # for thing in p1Betters:
    #     markedBoard[thing[0]][thing[1]] = "a"
    # for thing in p2Betters:
    #     markedBoard[thing[0]][thing[1]] = "b"
    #
    # for thing in TronP.board:
    #     print(thing)
    # print('\n')
    # for thing in markedBoard:
    #     print(thing)


    score = len(p1Betters)-len(p2Betters)
    print(TronP.player_to_move())
    if TronP.player_to_move() == 1:
        score *= -1
    # print("player")
    # print(TronP.player_to_move() + 1)
    # print("score")
    # print(score)
    # print('\n')
    # if score < 0:
    #     print((1-(abs(score)/numAvailSpaces)) * 0.5)
    #     return (1-(abs(score)/numAvailSpaces)) * 0.5
    print("SCORE")
    print(0.5 + 0.5*(score/numAvailSpaces))
    print("BOARD")
    for thing in TronP.board:
        print(thing)
    print('\n')
    return 0.5 + 0.5*(score/numAvailSpaces)




    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        pass

class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
