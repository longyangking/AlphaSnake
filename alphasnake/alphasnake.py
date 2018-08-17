import numpy as np 
import argparse

__version__ = "0.0.1"
__author__ = "Yang Long"
__info__ = "Play Snake Game with AI"

__default_board_shape__ = 15, 15
__default_state_shape__ = *__default_board_shape__, 3
__filename__ = 'model.h5'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__info__)
    parser.add_argument("--retrain", action='store_true', default=False, help="Re-Train AI")
    parser.add_argument("--train",  action='store_true', default=False, help="Train AI")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose")
    parser.add_argument("--playai", action='store_true', default=False, help="Play snake game with AI")
    parser.add_argument("--play", action='store_true', default=False, help="Play snake game")

    args = parser.parse_args()
    verbose = args.verbose

    if args.train:
        if verbose:
            print("Continue to re-train AI with state shape: {0}".format(__default_state_shape__))

        from ai import QAI
        from train_Qai import TrainAI

        ai = QAI(state_shape=__default_state_shape__, output_dim=5, verbose=verbose)
        if verbose:
            print("loading latest model: [{0}] ...".format(__filename__),end="")
        ai.load_nnet(__filename__)
        if verbose:
            print("load OK!")

        trainai = TrainAI(
            state_shape=__default_state_shape__,
            ai=ai,
            verbose=verbose
        )
        trainai.start(filename=__filename__)

        if verbose:
            print("The latest AI model is saved as [{0}]".format(__filename__))
       

    if args.retrain:
        if verbose:
            print("Start to re-train AI with state shape: {0}".format(__default_state_shape__))

        # Abondon the previous model and train a new one

        from train_Qai import TrainAI

        trainai = TrainAI(
            state_shape=__default_state_shape__,
            verbose=verbose
        )
        trainai.start(filename=__filename__)

        if verbose:
            print("The latest AI model is saved as [{0}]".format(__filename__))

    if args.play:
        print("Play snake game. Please close game in terminal after closing window (i.e, Press Ctrl+C).")
        from retrosnake import RetroSnake
        from retrosnake import Human

        Nx, Ny = __default_board_shape__
        retrosnake = RetroSnake(state_shape=__default_state_shape__)
        retrosnake.start()

    if args.playai:
        from ai import QAI
        from retrosnake import GameEngine

        ai = QAI(state_shape=__default_state_shape__, output_dim=5, verbose=verbose)
        if verbose:
            print("loading latest model: [{0}] ...".format(__filename__),end="")

        ai.load_nnet(__filename__)

        if verbose:
            print("load OK!")

        print("Play snake game. Please close game in terminal after closing window (i.e, Press Ctrl+C).")
        gameengine = GameEngine(state_shape=__default_state_shape__, player=ai, verbose=verbose)
        gameengine.start_ai()
        

        