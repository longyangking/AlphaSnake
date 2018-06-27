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
        '''
        Expand Leaf Node
        '''
        for action, prob in action_priors:
            if action not in self._childern:
                self._childern[action] = TreeNode(self,prob)

    def select(self,c_puct):
        '''
        Select node based on PUCT algorithm
        '''
        return max(self._childern.items(),
            key=lambda action_node: action_node[1].get_value(c_puct)
            )

    def update(self,leaf_value):
        '''
        Update node based on value
        '''
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        '''
        Update node and its parents' node based on value
        '''
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self,c_puct):
        '''
        Get PUCT value of node
        '''
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._childern == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    '''
    Monte Carlo Tree Search: Improver of deep learning model
    '''
    def __init__(self, evaluate_function, c_puct, n_playout, verbose=False):
        self.root = TreeNode(None, 1.0)
        self.evaluate_function = evaluate_function
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.verbose = verbose

    def search(self, engine, temperature):
        '''
        Search best action based on MCTS simulations
        '''
        for i in range(self.n_playout):
            _engine = engine.clone()
            if self.root.is_leaf():
                # Expand and Evaluate
                state = engine.get_state()
                probs, value = self.evaluate_function(state)

                actions = np.arange(5)
                self.root.expand(zip(actions, probs))

                # Backup
                self.root.update(value)
            else:
                # Select 
                action, node = self.root.select(self.c_puct)
                _engine.play(action)
                while not node.is_leaf():
                    action, node = node.select(self.c_puct)
                    _engine.play(action)

                # Expand and Evaluate
                state = _engine.get_state()
                probs, value = self.evaluate_function(state)

                actions = np.arange(5)
                node.expand(zip(actions, probs))

                # Backup
                node.update_recursive(value)

        # Play: Return simulation results
        actions_visits = [(action, node._n_visits) for action, node in self.root._childern.items()]
        actions, visits = zip(*actions_visits)
        action_probs = softmax(1.0/temperature*np.log(np.array(visits) + eps))
        return actions, action_probs

class NeuralNetwork:
    '''
    Deep learning neural network
    '''
    def __init__(self, 
        input_shape,
        output_shape,
        network_structure,
        learning_rate,
        l2_const,
        momentum
        ):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.network_structure = network_structure
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_const = l2_const

        self.model = None

    def init(self):
        state_tensor = Input(shape=self.input_shape, name='state_head')

        x = self.__conv_block(state_tensor, self.network_structure[0]['filters'], self.network_structure[0]['kernel_size'])
        if len(self.network_structure) > 1:
            for h in self.network_structure[1:]:
                x = self.__res_block(x, h['filters'], h['kernel_size'])
        
        action = self.__action_block(x)
        value = self.__value_block(x)
        
        self.model = Model(inputs=state_tensor, outputs=[action, value])
        self.model.compile(loss={'value_head': 'mean_squared_error', 'action_head': softmax_cross_entropy_with_logits},
			optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),	
			loss_weights={'value_head': 0.5, 'action_head': 0.5}	
			)

        return self.model

    def __conv_block(self, x, filters, kernel_size=3):
        '''
        Convolutional Neural Network
        '''
        out = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)
        return out

    def __res_block(self, x, filters, kernel_size=3):
        '''
        Residual Convolutional Neural Network
        '''
        out = Conv2D(
            filters = filters,
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
        out = Conv2D(
            filters = 32,
            kernel_size = (3,3),
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)

        out = Flatten()(out)
        out = Dense(
            36,
            use_bias=False,
            activation='linear',
            kernel_regularizer= regularizers.l2(self.l2_const)
		)(out)
        out = LeakyReLU()(out)

        value = Dense(
			1, 
            use_bias=False,
            activation='tanh',
            kernel_regularizer=regularizers.l2(self.l2_const),
            name = 'value_head'
			)(out)

        return value

    def __action_block(self, x):
        out = Conv2D(
            filters = 32,
            kernel_size = (3,3),
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)

        out = Flatten()(out)
        out = Dense(
            36,
            use_bias=False,
            activation='linear',
            kernel_regularizer= regularizers.l2(self.l2_const)
		)(out)
        out = LeakyReLU()(out)

        action = Dense(
			self.output_shape, 
            use_bias=False,
            activation='tanh',
            kernel_regularizer=regularizers.l2(self.l2_const),
            name = 'action_head'
			)(out)

        return action
        
class AI:
    '''
    Artificial Intelligence Player
    '''
    def __init__(self,
        state_shape,
        action_types=5,
        network_structure=None,
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

        if network_structure is not None:
            self.network_structure = network_structure
        else:
            # Default Neural Network Structure
            self.network_structure = list()
            self.network_structure.append({'filters':32, 'kernel_size':3})
            self.network_structure.append({'filters':32, 'kernel_size':3})
            self.network_structure.append({'filters':32, 'kernel_size':3})
            self.network_structure.append({'filters':32, 'kernel_size':3})

        self.model = None

    def init(self):
        '''
        Initiate the learning model
        '''
        self.model = NeuralNetwork(
            input_shape = self.state_shape,
            output_shape = self.action_types,
            network_structure = self.network_structure
        ).init()

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

    def plot_model(self, filename='model.png'):
        from keras.utils import plot_model
        plot_model(self.model, show_shapes=True, to_file=filename)

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

    def evaluate_function(self, state):
        '''
        Evaluate status based on current state information
        '''
        state = np.array(state).reshape(1, *state_shape)
        action, value = self.model.predict(state)
        return action[0], value[0]

    def play(self, engine, s_mcts=True):
        '''
        Play game according to the information from game engine
        '''
        n_simulations = 10  # Number of MCTS simulations
        c_puct = 0.95   
        n_playout = 10  # Depth of Monte Carlo Tree (Number of playout)

        if s_mcts:
            # Run MCTS simulations
            n_actions = np.zeros(5)
            for i in range(self.n_simulations):
                _engine = engine.clone()
                mcts = MCTS(
                    evaluate_function=self.evaluate_function, 
                    c_puct=c_puct, 
                    n_playout=n_playout, 
                    verbose=self.verbose)
                )
                mcts_actions, probs = mcts.search(engine=_engine)
                action = np.random.choice(
                    mcts_actions,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                n_actions[action] += 1
            action_index = np.argmax(n_actions)
        else:
            # Evaluate state directly
            state = engine.get_state()
            probs, value = self.evaluate_function(state=state)
            actions = np.arange(5)
            action_index = np.random.choice(
                actions,
                p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )  
        
        action = np.zeros(5)
        action[action_index] = 1
        return action

if __name__ == "__main__":
    print("Just for debug, not main parts")
    network_structure = list()
    network_structure.append({'filters':20, 'kernel_size':3})
    network_structure.append({'filters':20, 'kernel_size':3})
    network_structure.append({'filters':20, 'kernel_size':3})
    network_structure.append({'filters':20, 'kernel_size':3})

    input_shape = (10,10,1)
    output_shape = 5

    nn = NeuralNetwork(
        input_shape=input_shape,
        output_shape=output_shape,
        network_structure=network_structure,
        learning_rate=1e-4,
        l2_const=1e-4,
        momentum=0.9).init()

    print(nn.summary())