import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TM(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(TM, self).__init__()
        self.num_actions = num_actions


        self.learning_rate = 0.0005
        self.gamma = 0.99

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # self.model_output = tf.keras.Sequential()
        self.DL1 = tf.keras.layers.Dense(3*state_size, activation='relu')
        self.DL2 = tf.keras.layers.Dense(num_actions)

        self.hidden_state_size = 100
        self.DL3 = tf.keras.layers.Dense(self.hidden_state_size, activation='relu')
        self.DL4 = tf.keras.layers.Dense(1)

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        states_as_tensor = tf.dtypes.cast(tf.convert_to_tensor(states), tf.int32)
        dl1Output = self.DL1(states_as_tensor)

        dl2Output = self.DL2(dl1Output)
        return tf.nn.softmax(dl2Output)

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        # TODO: implement this :D
        crit1 =  self.DL3(tf.convert_to_tensor(states))
        crit2 = self.DL4(crit1)
        return crit2

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        Remember that the loss is similar to the loss as in part 1, with a few specific changes.

        1) In your actor loss, instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage.
        See handout/slides for definition of advantage.

        2) In your actor loss, you must use tf.stop_gradient on the advantage to stop the loss calculated on the actor network
        from propagating back to the critic network.

        3) See handout/slides for how to calculate the loss for your critic network.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """

        if(len(states) == 0):
            return tf.cast(0, tf.float32)
        probabilities = self.call(states)

        actionsPairedWithActionNumber = tf.concat( [tf.reshape(tf.range(len(states)), (len(states), 1)), tf.reshape(tf.squeeze(actions), (len(actions), 1))], 1)
        probabilitiesForEachAction = tf.gather_nd(probabilities, actionsPairedWithActionNumber)

        actorLoss = tf.reduce_sum(tf.math.multiply(tf.math.log(probabilitiesForEachAction),tf.stop_gradient(tf.math.subtract(self.value_function(states), discounted_rewards))))
        criticLoss = tf.reduce_sum(tf.math.square(tf.math.subtract(self.value_function(states), discounted_rewards)))
        return actorLoss + criticLoss
