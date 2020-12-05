import TronModel
import tensorflow as tf
import random

def trainOneGame(TronP, model):
    p1_sar, p2_sar = generate_trajectory(TronP, model)

    p1_discounted_rewards = discount(p1_sar["rewards"])
    p2_discounted_rewards = discount(p2_sar["rewards"])


    with tf.GradientTape() as tape:

        p1_loss = model.loss(p1["states"], p1["actions"], p1_discounted_rewards)
        p2_loss = model.loss(p2["states"], p2["actions"], p2_discounted_rewards)

        p1_gradients = tape.gradient(loss, model.trainable_variables)
        p2_gradients = tape.gradient(loss, model.trainable_variables)

        model.optimizer.apply_gradients(zip(p1_gradients, model.trainable_variables))
        model.optimizer.apply_gradients(zip(p2_gradients, model.trainable_variables))
    return tf.reduce_sum(rewards)

def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
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
    state = env.reset()
    done = False

    while not done:
        # TODO:
        # 1) use model to generate probability distribution over next actions
        # 2) sample from this distribution to pick the next action
        distrib = model.call(tf.expand_dims(state, axis=0))

        action = np.random.choice(len(tf.squeeze(distrib)), 1, p=tf.squeeze(distrib).numpy())[0]


        states.append(state)
        actions.append(action)
        state, rwd, done, _ = env.step(action)
        rewards.append(rwd)

    return states, actions, rewards

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


def main():

    tm = TronModel()
    mapList = []
    for i in range(1000):
        mapChoice = random.choice(mapList)
        trainOneGame(TronProblem(mapChoice, 0))
    tm.save("trainedModel")








if __name__ == '__main__':
    main()
