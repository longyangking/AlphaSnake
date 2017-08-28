import sys
sys.path.append("..")

import alphasnake
import time
import numpy as np


if __name__ == "__main__":
    area = np.zeros((10,10))
    # Snake
    area[2,2] = 1
    area[3,2] = 1
    area[4,2] = 1
    area[5,2] = 1
    area[5,3] = 1

    # Food
    area[6,6] = -1

    sizeunit = 30
    ui = alphasnake.ui.UI(pressaction=lambda x:x,area=area,sizeunit=sizeunit)
    ui.start()

