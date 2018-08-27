import numpy as np 
from ai import QAI
import time
import random
from gameutils import Snake, GameZone
from collections import deque

class SelfplayEngine:
    '''
    Self-play Engine for AI model
    '''
    def __init__(self, ai, verbose):
        self.ai = ai

        self.Nx, self.Ny, self.channel = ai.get_state_shape()
        self.verbose = verbose

        # training data
        self.states = list()
        self.areas = list()
        self.scores = list()
        self.actions = list()
    
    def update_states(self):
        '''
        Update stored states
        '''
        state = np.zeros((self.Nx, self.Ny, self.channel))
        n_areas = len(self.areas)
        for i in range(self.channel):
            if i+1 <= n_areas:
                state[:,:,-(i+1)] = self.areas[-(i+1)]

        self.states.append(state)

    def get_state(self):
        '''
        Get the state matrix
        '''
        if len(self.states) == 0:
            return np.zeros((self.Nx, self.Ny, self.channel))
        else:
            return self.states[-1]

    def start(self):
        '''
        Start a game for AI model
        '''
        area_shape = (self.Nx, self.Ny)
        gamezone = GameZone(
            area_shape=area_shape,
            player=self.ai,
            is_selfplay=True
        )

        gamezone.start(snakelength=5)

        # Main process
        flag = True
        while flag:
            self.areas.append(gamezone.get_area())
            self.update_states()

            flag, action = gamezone.update(self.get_state(), epsilon=0.8)

            self.actions.append(action)
            self.scores.append(gamezone.get_score())

        states = np.array(self.states)

        states_next = np.copy(states)
        states_next[:-1] : states_next[1:]
        states_next[-1] = np.zeros_like(states_next[-1])

        scores = np.array(self.scores).reshape(-1,1)
        actions = np.array(self.actions).reshape(-1,1)

        terminals = np.zeros(len(self.scores)).reshape(-1,1)
        terminals[-1] = 1

        data = (states, actions, scores, states_next, terminals)
        return data

class TrainAI:
    '''
    Train AI model process
    '''
    def __init__(self, 
        state_shape,
        replay_size=10000,
        ai=None,
        verbose=False
    ):

        self.state_shape = state_shape
        self.verbose = verbose

        if ai is None:
            self.ai = QAI(
                state_shape=self.state_shape,
                output_dim=5,
                verbose=self.verbose
                )
        else:
            self.ai = ai

        self.replay_size = replay_size
        self.dataset = deque()

    def get_selfplay_data(self, n_round):
        '''
        Run self-play and then return the results
        '''
        if self.verbose:
            starttime = time.time()
            count = 0

        for i in range(n_round):
            if self.verbose:
                print("Start self-playing to obtain data...(round {0})".format(i+1))

            engine = SelfplayEngine(
                ai=self.ai,
                verbose=self.verbose
            )
            data = engine.start()
            states, actions, rewards, states_next, terminals = data

            for j in range(len(terminals)):
                self.dataset.append((states[j], actions[j], rewards[j], states_next[j], terminals[j]))
                if len(self.dataset) > self.replay_size:
                    self.dataset.popleft()

            count += len(terminals)
                
        if self.verbose:
            endtime = time.time()
            print("End of self-play: Run Time {0:.2f}s, Set Size: {1}".format(endtime-starttime, count))

    def update_ai(self, minibatch_size):
        '''
        Evaluate the performance of AI player and return score value
        '''
        if self.verbose:
            print("Updating neural network of AI model ...")

        minibatch = random.sample(self.dataset, minibatch_size)
        loss = self.ai.train_on_batch(minibatch)

        if self.verbose:
            print("End of updation with loss: {0:.4f}".format(loss))

        return loss

    def start(self, filename):
        '''
        Main process of training AI
        '''
        
        n_round = 10
        n_epochs = 100000
        replay_size = 10000
        minibatch_size = 32
        verbose_interval = 1
        save_interval = 10

        for i in range(n_epochs):
            if self.verbose:
                if (i+1)%verbose_interval == 0:
                    print("Train Batch: {0}".format(i+1))
            
            self.get_selfplay_data(n_round)
            loss = self.update_ai(minibatch_size)

            if self.verbose:
                print("Saving model...",end="")

            if (i+1)%save_interval == 0:
                self.ai.save_nnet(filename)

            if self.verbose:
                print("OK!")

            