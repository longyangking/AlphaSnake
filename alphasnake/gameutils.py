import numpy as np 

class Snake:
    def __init__(self,head,body,direction,velocity,player):
        self.head = head
        self.body = body
        self.direction = direction
        self.player = player
        self.velocity = velocity

    def getbody(self):
        return self.body

    def gethead(self):
        return self.head

    def growup(self):
        self.body.insert(0,self.head)

    def survive(self):
        for i in range(2,len(self.body)):
            if (self.head[0] == self.body[i][0]) and (self.head[1] == self.body[i][1]):
                return False
        return True

    def update(self, state):
        (Nx,Ny) = area.shape
    
        # direction, action = self.player.play(
        #     head=self.head,
        #     body=self.body,
        #     area=area)  
        
        action = self.player.play(state) 

        direction = self.direction
        if action == 1:
            direction = (1,0)
        elif action == 2:
            direction = (-1,0)
        elif action == 3:
            direction = (0,1)
        elif action == 4:
            direction = (0,-1)

        # snake never ture around
        if (direction[0] == -self.direction[0]) and (direction[1] == -self.direction[1]):
            direction = self.direction

        x = (self.head[0] + direction[0])%Nx
        y = (self.head[1] + direction[1])%Ny

        
        self.direction = direction
            
        self.body.pop()
        self.head = (x,y)
        self.body.insert(0,self.head)

        return action

    def clone(self):
        newsnake = snake(
            head=self.head.copy(),
            body=self.body.copy(),
            direction=self.direction.copy(),
            player=self.player,
            velocity=self.velocity.copy()
        )
        return newsnake

class GameZone:
    def __init__(self,area_shape,player,timeperiod=1.0,
            init_snake=None,
            init_score=0,
            init_area=None,
            init_resourcevalid=False,
            init_resourcepos=None,
            init_validplaces=None,
            init_record=None,
            is_selfplay=False   # The flag of self-play
            ):
        self.Nx, self.Ny = area_shape

        self.snake = None
        if init_snake is not None:
            self.snake = init_snake

        self.score = init_score
        self.player = player
        self.timeperiod = timeperiod

        self.resourcevalid = init_resourcevalid
        self.resourcepos = init_resourcepos

        self.area = np.zeros(zone_shape)
        if init_area is not None:
            self.area = init_area

        self.validplaces = list(range(self.Nx*self.Ny))
        if init_validplaces is not None:
            self.validplaces = init_validplaces

        self.record = Record(Nx=self.Nx, Ny=self.Ny)
        if init_record is not None:
            self.record = init_record
    
        self.is_selfplay = is_selfplay

    def get_area(self):
        '''
        Get area info:

        0: blank
        -1: resource
        1: snake
        '''
        return np.copy(self.area)
    
    def get_score(self):
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

        self.snake = Snake(head=head,body=body,direction=direction,
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

    def update(self, state):
        if not self.resourcevalid:
            self.setresource()

        starttime = time.time()
        action = self.snake.update(state)
        endtime = time.time()
        time.sleep(self.timeperiod-(endtime-starttime))

        self.updatearea()   # Update Area
        flag = self.snake.survive()

        return flag, action
        