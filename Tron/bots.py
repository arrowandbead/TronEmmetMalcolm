#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
import tensorflow as tf

# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""

    def __init__(self):

        self.model = tf.keras.models.load_model("trainedModel")

        self.move_map = {
        0 : "U",
        1 : "D",
        2 : "L",
        3 : "R"
        }

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        result = self.model(self.parse(asp))

        best_move = tf.argmax(result)

        return self.move_map[result]

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

        max_board_size = 5
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
