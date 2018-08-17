import numpy as np
from gameutils import GameZone
from ui import UI

class Human:
    def __init__(self):
        self.direction = (1,0)
        self.action = 0

    def setdirection(self,mode):
        if mode == 1:
            self.direction = (1,0)
        elif mode == 2:
            self.direction = (-1,0)
        elif mode == 3:
            self.direction = (0,1)
        elif mode == 4:
            self.direction = (0,-1)
        
        self.action = mode

    def play(self, state):
        return self.action

class GameEngine:
    def __init__(self, state_shape, player, timeperiod=0.5, verbose=False):
        self.state_shape = state_shape
        self.channel = self.state_shape[-1]
        self.player = player
        self.ui = None
        self.pause = False

        self.timeperiod = timeperiod

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
        state = np.zeros(self.state_shape)
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
            return np.zeros(self.state_shape)
        else:
            return self.states[-1]

    def pause_game(self,mode):
        if mode == 100:
            if self.pause:
                self.pause = False

                if self.verbose:
                    print("Start game.")   
            else:
                self.pause = True

                if self.verbose:
                    print("Pause game.")   

    def start_ai(self):
        gamezone = GameZone(area_shape=self.state_shape[:2],player=self.player,timeperiod=self.timeperiod)
        gamezone.start()

        sizeunit = 30
        area = gamezone.get_area()
        ui = UI(pressaction=self.pause_game,area=area,sizeunit=sizeunit)
        ui.start()
        
         # Main process
        flag = True
        while flag:
            if not self.pause:
                self.areas.append(gamezone.get_area())
                self.update_states()

                flag, action = gamezone.update(self.get_state(), epsilon=0.8)

                self.actions.append(action)
                self.scores.append(gamezone.get_score())
                ui.setarea(area=gamezone.get_area())

        ui.gameend(gamezone.get_score())

    def start(self):
        gamezone = GameZone(area_shape=self.state_shape[:2],player=self.player,timeperiod=self.timeperiod)
        gamezone.start()

        sizeunit = 30
        area = gamezone.get_area()
        ui = UI(pressaction=self.player.setdirection,area=area,sizeunit=sizeunit)
        ui.start()
        
         # Main process
        flag = True
        while flag:
            self.areas.append(gamezone.get_area())
            self.update_states()

            flag, action = gamezone.update(self.get_state(), epsilon=0.8)

            self.actions.append(action)
            self.scores.append(gamezone.get_score())
            ui.setarea(area=gamezone.get_area())

        ui.gameend(gamezone.get_score())

class RetroSnake:
    def __init__(self, state_shape):
        self.gameengine = GameEngine(state_shape=state_shape,player=Human())

    def start(self):
        self.gameengine.start()

if __name__=="__main__":
    # Just for debugging
    retrosnake = RetroSnake()
    retrosnake.start()