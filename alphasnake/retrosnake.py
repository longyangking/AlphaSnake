import gameengine
import ui
import player

class GameEngine:
    def __init__(self,Nx,Ny):
        self.Nx = Nx
        self.Ny = Ny
        self.gameengine = None
        self.player = player.Human()
        self.ui = None

    def start(self):
        self.gameengine = gameengine.GameEngine(Nx=self.Nx,Ny=self.Ny,player=self.player,timeperiod=0.5)
        self.gameengine.start()

        sizeunit = 30
        area = self.gameengine.get_area()
        self.ui = ui.UI(pressaction=self.player.setdirection,area=area,sizeunit=sizeunit)
        self.ui.start()
        
        while self.gameengine.update():
            self.ui.setarea(area=self.gameengine.get_area())

        self.ui.gameend(self.gameengine.get_score())

class RetroSnake:
    def __init__(self, Nx=15, Ny=15):
        self.gameengine = GameEngine(Nx,Ny)

    def start(self):
        self.gameengine.start()

if __name__=="__main__":
    # Just for debugging
    retrosnake = RetroSnake()
    retrosnake.start()