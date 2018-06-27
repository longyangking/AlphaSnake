import numpy as np 

class snake:
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

    def update(self,area):
        (Nx,Ny) = area.shape
    
        direction, action = self.player.play(
            head=self.head,
            body=self.body,
            area=area)  

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