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
        if action[1]:
            self.direction = (1,0)
        if action[2]:
            self.direction = (-1,0)
        if action[3]:
            self.direction = (0,1)
        if action[4]:
            self.direction = (0,-1)

        return self.direction, action

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

        data = gameengine.get_data()
        data = self.__geometry_operators(data)

        return data

    def __geometry_operators(self, data):
        '''
        Reflection or Rotation of training data to remove geometrical or coordination dependence
        '''

        # TODO Reflection or Rotation

        return data

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
        datasets = engine.start()
        return datasets

    def evaluate_ai(self):
        '''
        Evaluate the performance of AI player and return score value
        '''

        # TODO Evaluation process

        value = 0
        return value

    def start(self):
        '''
        Main process of training AI
        '''
        temperature = 1.0
        n_selfplay = 100
        n_epochs = 1000
        interval_save = 10
        interval_evaluate = 10

        value = 0
        for i in range(n_epochs):
            if self.verbose:
                print("Train Batch: {0}".format(i+1))
            
            if self.verbose:
                print("Start self-playing")
            selfplay_data = self.get_selfplay_data()
            if self.verbose:
                print("End self-playing")

            if self.verbose:
                print("Updating ai...",end="")
            self.ai.update(selfplay_data)
            if self.verbose:
                print("OK!")

            if (i+1)%interval_save == 0:
                self.ai.save('selftrain_{0}.h5'.format(i+1))

            if (i+1)%interval_evaluate == 0:
                new_value = self.evaluate_ai()
                if new_value > value:
                    self.ai.save('model.h5')

            