import numpy as np 

class snake:
    def __init__(self,head,body,direction,player):
        self.head = head
        self.body = body
        self.direction = direction
        self.player = player

    def getbody(self):
        return self.body

    def gethead(self):
        return self.head

    def growup(self):
        self.body.insert(0,self.head)

    def update(self,area):
        self.direction = self.player.play(head=head,body=body,area=area,player=self.player)

        self.body.pop()
        x = head[0] + self.direction[0]
        y = head[0] + self.direction[1]

        self.head = (x,y)
        self.body.insert(0,self.head)