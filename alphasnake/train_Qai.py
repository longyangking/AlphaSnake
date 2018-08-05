import numpy as np 
from ai import QAI
import time
import random
from gameutils import Snake, GameZone

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
        pass

    def get_state(self):
        '''
        Get the state matrix
        '''
        if len(self.states) > 0:
            return np.zeros((self.Nx, self.Ny, self.channel))
        else:
            return self.states[-1]

    def start(self):
        '''
        Start a game for AI model
        '''
        if self.verbose:
            starttime = time.time()
            print("Self-playing...", end="")

        area_shape = (self.Nx, self.Ny)
        gamezone = GameZone(
            area_shape=area_shape,
            player=self.ai
        )

        gamezone.start()

        # Main process
        flag = True
        while flag:
            self.areas.append(gamezone.get_area())
            self.update_states()

            flag, action = gamezone.update(self.get_state())

            self.actions.append(action)
            self.scores.append(self.gamezone.get_score())

        if self.verbose:
            endtime = time.time()
            print("End: Run Time {0:.2f}s".format(endtime-starttime))

        states = np.array(self.states)
        scores = np.array(self.scores).reshape(-1,1)
        actions = np.array(self.actions).reshape(-1,1)

        terminals = np.zeros(len(self.scores)).reshape(-1,1)
        terminals[-1] = 1

        data = zip(states, actions, scores, terminals)
        return data

class TrainAI:
    '''
    Train AI model process
    '''
    def __init__(self, 
        state_shape,
        verbose=False
    ):

        self.state_shape = state_shape
        self.verbose = verbose

        self.ai = AI(
            state_shape=self.state_shape,
            action_types=5,
            verbose=self.verbose
            )

    def get_selfplay_data(self, n_round):
        '''
        Run self-play and then return the results
        '''
        all_states, all_actions, all_rewards, all_terminals = list(), list(), list(), list()

        if self.verbose:
            print("Start self-playing to obtain data...")

        for i in range(n_round):
            engine = SelfplayEngine(
                ai=self.ai,
                verbose=self.verbose
            )
            data = engine.start()
            states, actions, rewards, terminals = data

            for i in range(len(terminals)):
                all_states.append(states[i])
                all_actions.append(actions[i])
                all_rewards.append(rewards[i])
                all_terminals.append(terminals[i])

        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        all_rewards = np.array(all_rewards)
        all_terminals = np.array(all_terminals)
            
        dataset = zip(all_states, all_actions, all_rewards, all_terminals)

        if self.verbose:
            print("End of self-play. The shape of dataset: [states: {0}; actions: {1}; rewards: {2}; terminals: {3}]".format(
                all_states.shape,
                all_actions.shape,
                all_rewards.shape,
                all_terminals.shape
            ))

        return dataset

    def update_ai(self, dataset, minibatch_size):
        '''
        Evaluate the performance of AI player and return score value
        '''
        minibatch = random.sample(dataset, minibatch_size)
        loss = self.ai.train_on_batch(minibatch)
        return loss

    def start(self):
        '''
        Main process of training AI
        '''
        
        n_round = 30
        n_epochs = 1000

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

            