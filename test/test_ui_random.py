import sys
sys.path.append("..")

import alphasnake
import time
import numpy as np


if __name__ == "__main__":
    area = np.random.randint(-1,2,size=(10,10))
    sizeunit = 30
    ui = alphasnake.ui.UI(pressaction=lambda x:x,area=area,sizeunit=sizeunit)
    ui.start()

    for _ in range(2):
        time.sleep(0.5)
        ui.setarea(area=np.random.randint(-1,2,size=(10,10)))
    ui.gameend(10)
