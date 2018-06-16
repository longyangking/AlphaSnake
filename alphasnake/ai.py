from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, Add
from keras.optimizers import SGD
from keras import regularizers
import keras.backend as K
import tensorflow as tf

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only error will be shown

def softmax_cross_entropy_with_logits(y_true, y_pred):
    '''
    Softmax Cross Entropy Function with logits
    '''
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
	return loss

class TreeNode:
    '''
    Monte Carlo Tree
    '''
    def __init__(self,parent,prior_p):
        self._parent = parent
        self._childern = {} # Save childre nodes in Hash data structure
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self,action_priors):
        #print(action_priors)
        for action, prob in action_priors:
            if action not in self._childern:
                self._childern[action] = TreeNode(self,prob)

    def select(self,c_puct):
        return max(self._childern.items(),
            key=lambda action_node: action_node[1].get_value(c_puct)
            )

    def update(self,leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self,c_puct):
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._childern == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    '''
    Monte Carlo Tree Search
    '''
    def __init__(self, value_function, c_puct, n_playout, verbose=False):
        self._root = TreeNode(None, 1.0)
        self._value_function = value_function
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.role = role
        self.verbose = verbose

    def playout(self):
        '''
        Play out
        '''
        pass

    def update_tree(self):
        '''
        Update Monte Carlo Tree after move
        '''
        pass 

class NeuralNetwork:
    def __init__(self, 
        input_shape,
        output_shape,
        network_structure,
        learning_rate,
        momentum
        ):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.network_structure = network_structure
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.model = None
        self.init()

        return self.model

    def init(self):
        state_tensor = Input(shape=self.input_shape, name='state_head')

        # TODO construct neural network
        
        action = self.__action_block()
        value = self.__value_block()
        
        self.model = Model(input=state_tensor, output=[action, value])
        self.model.compile(loss={'value_head': 'mean_squared_error', 'action_head': softmax_cross_entropy_with_logits},
			optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),	
			loss_weights={'value_head': 0.5, 'action_head': 0.5}	
			)

    def __conv_block(self, x, filters, kernel_size=3):
        out = Conv2D(
            filters = nb_filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)
        return out

    def __res_block(self, x, filters, kernel_size=3):
        out = Conv2D(
            filters = nb_filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = Add()([out, x])
        out = LeakyReLU()(out)
        return out

    def __value_block(self, x):
        pass

    def __action_block(self, x):
        pass

class AI:
    def __init__(self,
        state_shape,
        action_types=5,
        learning_rate=2e-3,
        lr_multiplier=1.0,
        l2_const=1e-4,
        verbose=False
    ):

        self.state_shape = state_shape
        self.action_types = action_types

        self.learning_rate = learning_rate
        self.lr_multiplier = lr_multiplier
        self.l2_const = l2_const
        self.verbose = verbose

        network_structure = list()
        network_structure.append({'filters':20, 'kernel_size':3})
        network_structure.append({'filters':20, 'kernel_size':3})
        network_structure.append({'filters':20, 'kernel_size':3})
        network_structure.append({'filters':20, 'kernel_size':3})

        self.model = None

    def init(self):
        '''
        Initiate the learning model
        '''
        self.model = NeuralNetwork(
            input_shape = self.state_shape,
            output_shape = self.action_types,
            network_structure = network_structure
        )

    def load_model(self, filename):
        '''
        Load Model with file name
        '''
        from keras.models import load_model as LOAD_MODEL
        self.model = LOAD_MODEL(filename)

    def save_model(self, filename):
        '''
        Save model with file name
        '''
        self.model.save(filename)

    def update(self, data):
        '''
        Update neural network
        '''
        Xs, ys = zip(*data)
        states = Xs
        actions, values = zip(*ys)
        self.model.train_on_batch(states, [actions, values])

    def train(self, data, batchsize=128, epochs=30, validation_split=0.0, verbose=False):
        '''
        Train neural network
        '''
        Xs, ys = zip(*data)
        states = Xs
        actions, values = zip(*ys)
        self.model.fit(
            states, [actions, values], 
            epochs=epochs, 
            batchsize=batchsize, 
            validation_split=validation_split, 
            verbose=verbose
        )

    def play(self, state):
        '''
        Play game based on state information
        '''
        state = np.array(state).reshape(1, *state_shape)
        action, value = self.model.predict(state)
        return action[0], value[0]