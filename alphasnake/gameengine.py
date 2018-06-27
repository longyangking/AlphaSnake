import numpy as np 
import snake
import time
import copy

class GameEngine:
    def __init__(self,Nx,Ny,player,timeperiod=1.0,
            init_snake=None,
            init_score=0,
            init_area=None,
            init_resourcevalid=False,
            init_resourcepos=None,
            init_validplaces=None,
            init_record=None
            ):
        self.Nx = Nx
        self.Ny = Ny

        self.snake = None
        if init_snake is not None:
            self.snake = init_snake

        self.score = init_score
        self.player = player
        self.timeperiod = timeperiod

        self.resourcevalid = init_resourcevalid
        self.resourcepos = init_resourcepos

        self.area = np.zeros((Nx,Ny))
        if init_area is not None:
            self.area = init_area

        self.validplaces = list(range(self.Nx*self.Ny))
        if init_validplaces is not None:
            self.validplaces = init_validplaces

        self.record = Record(Nx=self.Nx, Ny=self.Ny)
        if init_record is not None:
            self.record = init_record
    
    def getarea(self):
        return self.area.copy()
    
    def getscore(self):
        return self.score
        
    def __pos2place(self,pos):
        place = self.Nx*pos[1] + pos[0]
        return place

    def __place2pos(self,place):
        pos = (place%self.Nx,int(place/self.Nx))
        return pos

    def initsnake(self,snakelength,velocity):
        # Initiate the retro snake
        x = np.random.randint(self.Nx)
        y = np.random.randint(self.Ny)
        head = (x,y) 

        choice = np.random.randint(4)
        direction = (1,0)
        if choice == 0:
            direction = (1,0)
        elif choice == 1:
            direction = (-1,0)
        elif choice == 2:
            direction = (0,1)
        else:
            direction = (0,-1)

        body = list()
        body.append(head)
        for i in range(snakelength-1):
            x = (x - direction[0])%self.Nx
            y = (y - direction[1])%self.Ny
            body.append((x,y))

        # Update Area Information
        for pos in body:
            self.area[pos] = 1
            self.validplaces.remove(self.__pos2place(pos))  #not valid place anymore

        self.snake = snake.snake(head=head,body=body,direction=direction,
                velocity=velocity,player=self.player)

    def start(self,snakelength=3,velocity=1):
        self.area = np.zeros((self.Nx,self.Ny))
        self.resourcevalid = False
        self.validplaces = list(range(self.Nx*self.Ny))
        self.initsnake(snakelength=snakelength,velocity=velocity)
        
        return True

    def restart(self,snakelength=3,velocity=1):
        return self.start(snakelength=snakelength,velocity=velocity)

    def setresource(self):
        choice = np.random.randint(len(self.validplaces))
        place = self.validplaces[choice]
        pos = self.__place2pos(place)

        self.resourcepos = pos
        self.area[pos] = -1
        self.validplaces.remove(place)

        self.resourcevalid = True

    def updatearea(self):
        body = self.snake.getbody()
        head = self.snake.gethead()

        if self.area[head] == -1:
            self.score += 1
            self.snake.growup()
            self.resourcevalid = False

        # update area and valid places
        self.validplaces = list(range(self.Nx*self.Ny))
        self.area = np.zeros((self.Nx,self.Ny))
        for pos in body:
            self.area[pos] = 1
            place = self.__pos2place(pos)
            if place in self.validplaces:
                self.validplaces.remove(place)

        if self.resourcevalid:
            self.area[self.resourcepos] = -1

    def update(self):
        if not self.resourcevalid:
            self.setresource()

        starttime = time.time()
        action = self.snake.update(self.area)
        endtime = time.time()
        time.sleep(self.timeperiod-(endtime-starttime))

        self.updatearea()   # Update Area
        flag = self.snake.survive()

        # Record Game
        score = flag*self.getscore()
        self.record.append(area=self.getarea(), action=action, score=self.getscore())

        return flag

    def get_state(self, state_len=3):
        state = self.record.get_state(state_len=state_len)
        return state

    def clone(self):
        engine = GameEngine(
            Nx=self.Nx,
            Ny=self.Ny,
            player=self.player,
            timeperiod=self.timeperiod,

            init_snake=self.snake.clone(),
            init_score=self.getscore(),
            init_area=self.getarea(),
            init_resourcevalid=self.resourcevalid,
            init_resourcepos=self.resourcepos.copy(),
            init_validplaces=self.validplaces.copy(),
            init_record=self.record.clone()
            )
        return engine
        
    def get_data(self, state_len=3):
        states, actions, values = self.record.get_data(state_len=state_len)
        Xs = states
        ys = zip(actions, values)
        data = zip(Xs,ys)
        return data

class Record:
    def __init__(self, Nx, Ny, 
            init_areas=None,
            init_actions=None,
            init_values=None):
        self.Nx, self.Ny = Nx, Ny

        self.areas = list()
        if init_areas is not None:
            self.areas = init_areas

        self.actions = list()
        if init_actions is not None:
            self.actions = init_actions

        self.values = list()
        if init_values is not None:
            self.values = init_values

    def clone(self):
        init_areas = copy.deepcopy(self.areas)
        init_actions = copy.deepcopy(self.actions)
        init_values = copy.deepcopy(self.values)

        record = Record(
            Nx=self.Nx, Ny=self.Ny,
            init_areas=init_areas,
            init_actions=init_actions,
            init_values=init_values
            )
        return record

    def append(self, area, action, score):
        '''
        Append data and record
        '''
        value = score/self.Nx/self.Ny
        self.areas.append(area)
        self.actions.append(action)
        self.values.append(value)
        
    def get_state(self, state_len, bias=0):
        '''
        Get state
        '''
        n_areas = len(self.areas)
        state = np.zeros((self.Nx, self.Ny, state_len))
        if n_areas == 0:
            return state
        for i in range(state_len):
            if n_areas-1-i-bias >= 0:
                state[:,:,i] = self.areas[n_areas-1-i]
        return state

    def get_data(self, state_len):
        '''
        Get all data
        '''
        n_data = len(self.areas)
        states = list()
        for i in range(n_data):
            state = self.get_state(state_len, bias=i)
            states.append(state)
        values = np.array(self.values)
        actions = np.array(self.actions)

        return states, actions, values