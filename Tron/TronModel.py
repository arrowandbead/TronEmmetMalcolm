import os
import numpy as np
import tensorflow as tf
from bots import StudentBot

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TM(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        super(TM, self).__init__()


        self.learning_rate = 0.001
        self.gamma = 0.99

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.loss_function = tf.keras.losses.CategoricalCrossentropy()

        self.DL1 = tf.keras.layers.Dense(2890, activation='relu')

        self.CL1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.MPL1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.CL2 = tf.keras.layers.Conv2D(24, (2, 2), activation='relu', padding='same')
        self.MPL2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.flatten = tf.keras.layers.Flatten()

        self.DL2 = tf.keras.layers.Dense(289, activation='relu')

        self.outputLayer = tf.keras.layers.Dense(4)

        self.softmax = tf.keras.layers.Softmax()




    def call(self, state):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        states_as_tensor = tf.dtypes.cast(tf.convert_to_tensor(state), tf.int32)
        dl1Output = self.DL1(states_as_tensor)
        reshapedForConv = tf.reshape(dl1Output, [1,17,17,10])
        cl1Output = self.CL1(reshapedForConv)
        mpl1Output = self.MPL1(cl1Output)
        cl2Output = self.CL2(mpl1Output)
        mpl2Output = self.MPL2(cl2Output)
        flattened = self.flatten(mpl2Output)
        dl2Output = self.DL2(flattened)
        return self.softmax(self.outputLayer(dl2Output))
        



    # def value_function(self, states):
    #     """
    #     Performs the forward pass on a batch of states to calculate the value function, to be used as the
    #     critic in the loss function.
    #
    #     :param states: An [episode_length, state_size] dimensioned array representing the history of states
    #     of an episode.
    #     :return: A [episode_length] matrix representing the value of each state.
    #     """
    #     # TODO: implement this :D
    #     crit1 =  self.DL3(tf.convert_to_tensor(states))
    #     crit2 = self.DL4(crit1)
    #     return crit2

    def loss(self, p1states, p2states, p1parsedStates, p2parsedStates, TronP):

        numMove = {
        'U' : 0,
        'D' : 1,
        'L' : 2,
        'R' : 3
        }

        sb = StudentBot()
        p1RightActions = []
        p1Distrib = []


        for i in range(len(p1states)):
            TronP._start_state = p1states[i]
            correct = numMove[sb.decide(TronP)]
            p1RightActions.append(correct)
            parsedState = p1parsedStates[i]

            distrib = self.call(tf.expand_dims(parsedState, axis=0))
            distrib = tf.squeeze(distrib)
            p1Distrib.append(distrib)

        p1RightActionOneHot = tf.one_hot(p1RightActions, 4)
        p1Loss = self.loss_function(p1RightActionOneHot, p1Distrib)

        p2Loss = 0
        if(len(p2states) != 0):
            p2RightActions = []
            p2Distrib = []
            for i in range(len(p2states)):
                TronP._start_state = p2states[i]
                correct = numMove[sb.decide(TronP)]
                p2RightActions.append(correct)
                parsedState = p2parsedStates[i]
                distrib = self.call(tf.expand_dims(parsedState, axis=0))
                distrib = tf.squeeze(distrib)

                p2Distrib.append(distrib)


            p2RightActionOneHot = tf.one_hot(p2RightActions, 4)
            p2Loss = self.loss_function(p2RightActionOneHot, p2Distrib)

        # print(p1RightActionOneHot)
        # print(p2RightActionOneHot)
        # for thing in p1distrib:
        #     print(thing)
        # print("BREAK")
        # for thing in p2distrib:
        #     print(thing)
        if(p2Loss == 0):
            return p1Loss


        return (p1Loss + p2Loss)/2
