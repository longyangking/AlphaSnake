import numpy as np 
from ai import AI

class SelfplayEngine:
    '''
    Self-play Engine for AI model
    '''
    def __init__(self, ai, verbose):
        self.ai = ai
        self.verbose = verbose

    def start(self):
        '''
        Start a game for AI model
        '''
        n_mcts = 1000
        pass

    def __geometry_operators(self, data):
        '''
        Reflection or Rotation of training data to remove geometrical or coordination dependence
        '''
        pass

class TrainAI:
    '''
    Train AI model process
    '''
    def __init__(self, 
        state_shape=state_shape,
        verbose=verbose
    ):

        self.state_shape = state_shape
        self.verbose = verbose

        self.ai = AI(
            state_shape=self.state_shape,
            action_types=5,
            verbose=self.verbose
            )

    def get_selfplay_data(self):
        '''
        Run self-play and then return the results
        '''
        engine = SelfplayEngine(
            ai=self.ai,
            verbose=self.verbose
        )

    def start(self):
        temperature = 1.0
        n_selfplay = 100
        n_epochs = 1000

        for i in range(n_epochs):
            if self.verbose:
                print("Train Batch: {0}".format(i+1))

        pass