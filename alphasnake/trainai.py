import numpy as np 
from ai import AI
import time
from gameengine import GameEngine

class SelfPlayer:
    '''
    Player for self-play training process
    '''
    def __init__(self, ai):
        self.ai = ai 
        self.direction = (1,0)

    def play(self, engine, s_mcts=True):
        action = self.ai.play(engine, s_mcts)

        # Action == 0 means no change for move direction
        if action == 1:
            self.direction = (1,0)
        if action == 2:
            self.direction = (-1,0)
        if action == 3:
            self.direction = (0,1)
        if action == 4:
            self.direction = (0,-1)

        return self.direction

class SelfplayEngine:
    '''
    Self-play Engine for AI model
    '''
    def __init__(self, ai, Nx, Ny, verbose):
        self.player = SelfPlayer(ai=ai)

        self.Nx = Nx
        self.Ny = Ny
        self.verbose = verbose

    def start(self):
        '''
        Start a game for AI model
        '''
        n_mcts = 1000
        
        if self.verbose:
            starttime = time.time()
            print("Self-playing...", end="")

        gameengine = GameEngine(
            Nx=self.Nx,
            Ny=self.Ny,
            player=self.player,
            timeperiod=0.5,
            is_selfplay=True)
        gameengine.start()
        while gameengine.update():
            pass

        if self.verbose:
            endtime = time.time()
            print("End: Run Time {0:.2f}s".format(endtime-starttime))

        datasets = gameengine.get_datasets()
        datasets = self.__geometry_operators(datasets)

        return datasets

    def __geometry_operators(self, datasets):
        '''
        Reflection or Rotation of training data to remove geometrical or coordination dependence
        '''

        # TODO Reflection or Rotation

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