import TronModel
import tensorflow as tf
import random
from tronproblem import TronProblem
import numpy as np

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

    p1_visited_dict = {}
    p1_curr_loc = find_player(TronP.board, 1)
    p1_frontier = [p1_curr_loc,]
    p1_visited_dict[p1_curr_loc] = 0

    p2_visited_dict = {}
    p2_curr_loc = find_player(TronP.board, 2)
    p2_frontier = [p2_curr_loc,]
    p2_visited_dict[p2_curr_loc] = 0

    numAvailSpaces = 0
    setty = {"#", "-", 'x', '1', '2'}
    goody = {"*", "@", "!"}
    for thing in TronP.board:
        for b in thing:
            if b not in setty:
                numAvailSpaces += 1
            if b == "*" or b == "!":
                numAvailSpaces += 4
            if b == "@":
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

        while( len(p2_frontier) != 0):
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




    p1Set = p1_visited_dict.keys()
    p2Set = p2_visited_dict.keys()
    # print(p1Set)
    # print(p2Set)
    p1ScoreMod = 0
    tieScore = 0
    p1Betters = []
    for thing in p1Set:
        cellVal = TronP.board[thing[0]][thing[1]]
        if thing not in p2Set:
            if(cellVal == "@"):
                p1ScoreMod += 1
            elif(cellVal == "*" or cellVal == "!"):
                p1ScoreMod += 4
            p1Betters.append(thing)
        else:
            if(cellVal == "@"):
                tieScore += 2
            elif(cellVal == " "):
                tieScore += 1
            elif(cellVal == "*" or cellVal == "!"):
                tieScore += 5


    p2Betters =  []
    p2ScoreMod = 0
    for thing in p2Set:
        if thing not in p1Set:
            cellVal = TronP.board[thing[0]][thing[1]]
            if(cellVal == "@"):
                p2ScoreMod += 1
            elif(cellVal == "*" or cellVal == "!"):
                p2ScoreMod += 4

            p2Betters.append(thing)

    if TronP.player_to_move() == 0:
        p1ScoreMod += tieScore
    else:
        p2ScoreMod += tieScore

    score = (len(p1Betters) + p1ScoreMod) - (len(p2Betters) + p2ScoreMod)

    if TronP.player_to_move() == 1:
        score *= -1

    return 0.5 + 0.5*(score/numAvailSpaces)

def trainOneGame(TronP, model):
    p1_sar, p2_sar = generate_trajectory(TronP, model)

    p1_discounted_rewards = discount(p1_sar["rewards"])
    p2_discounted_rewards = discount(p2_sar["rewards"])


    with tf.GradientTape() as tape:

        p1_loss = model.loss(p1_sar["states"], p1_sar["actions"], p1_discounted_rewards)

        p1_gradients = tape.gradient(p1_loss, model.trainable_variables)

        model.optimizer.apply_gradients(zip(p1_gradients, model.trainable_variables))
    with tf.GradientTape() as tape:

        p2_loss = model.loss(p2_sar["states"], p2_sar["actions"], p2_discounted_rewards)

        p2_gradients = tape.gradient(p2_loss, model.trainable_variables)

        model.optimizer.apply_gradients(zip(p2_gradients, model.trainable_variables))


def generate_trajectory(TronP, model):
    move_map = {
        0 : "U",
        1 : "D",
        2 : "L",
        3 : "R"
    }

    p1Trajectory = {
        "states" : [],
        "actions" : [],
        "rewards" : []
    }
    p2Trajectory = {
        "states" : [],
        "actions" : [],
        "rewards" : []
    }
    state = TronP._start_state
    done = False

    while not done:

        playerMap = None
        ptm = state.player_to_move()
        if(state.player_to_move() == 0):
            playerMap = p1Trajectory
        else:
            playerMap = p2Trajectory


        parsedState = parse(state)

        distrib = model.call(tf.expand_dims(parsedState, axis=0))
        print("distribution")
        print(distrib)
        action = np.random.choice(len(tf.squeeze(distrib)), 1, p=tf.squeeze(distrib).numpy())[0]


        playerMap["states"].append(parsedState)
        playerMap["actions"].append(action)
        state = TronP.transition(state, move_map[action])
        done = TronP.is_terminal_state(state)
        rwd = None
        if(done):
            rwd = TronP.evaluate_state(state)[ptm]
        else:
            rwd = eval_func(state)
        playerMap["rewards"].append(rwd)

    return (p1Trajectory, p2Trajectory)

def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep.
    Refer to the slides to see how this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """



    outputList = [0]*len(rewards)
    for i in range(len(rewards)):
        if(i == 0):
            outputList[-1] = rewards[-1]
        else:
            outputList[-(i+1)] = rewards[-(i+1)] + discount_factor*outputList[-i]
    return outputList


def parse(TState):
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

    board = TState.board
    max_board_size = 17
    #board = TronP.board

    np_board = np.zeros(shape=(max_board_size, max_board_size), dtype=int)

    for i in range(len(board)):
        for j in range(len(board[0])):
            np_board[i][j] = featureNumMap[board[i][j]]

    np_board = np_board.flatten()
    onehot_board = np.eye(10, dtype=int)[np_board]
    flattened_onehot_board = onehot_board.flatten()

    non_board_state = np.zeros(shape=(11,), dtype=int)

    #Set ptm
    non_board_state[0] = int(TState.player_to_move())

    #Set player armors
    non_board_state[1] = TState.player_has_armor(0)
    non_board_state[2] = TState.player_has_armor(1)

    #Set player boost
    p1_boost = TState.get_remaining_turns_speed(0)
    if(p1_boost != 0):
        non_board_state[2+p1_boost] = 1

    p2_boost = TState.get_remaining_turns_speed(1)
    if(p2_boost != 0):
        non_board_state[6+p2_boost] = 1


    return np.concatenate( (flattened_onehot_board, non_board_state), axis=0)


def main():

    tm = TronModel.TM(2901, 4)
    mapList = ["center_block.txt", "diagonal_blocks.txt", "divider.txt", "empty_room.txt", "hunger_games.txt",  "joust.txt", "small_room.txt"]
    for i in range(200):
        if(i%10 == 0):
            print(i)
        mapChoice = random.choice(mapList)
        trainOneGame(TronProblem("maps/" + mapChoice, 0), tm)
    tm.save("trainedModel")








if __name__ == '__main__':
    main()
