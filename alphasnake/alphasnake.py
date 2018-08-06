import numpy as np 
import argparse

__version__ = "0.0.1"
__author__ = "Yang Long"
__info__ = "Play Snake Game with AI"

__default_board_shape__ = 15, 15
__default_state_shape__ = *__default_board_shape__, 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__info__)
    parser.add_argument("--retrain", action='store_true', default=False, help="Re-Train AI")
    parser.add_argument("--train",  action='store_true', default=False, help="Train AI")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose")
    parser.add_argument("--info", action='store_true', default=False, help="Show the process information")
    parser.add_argument("--playbyai", action='store_true', default=False, help="Play snake game with AI")
    parser.add_argument("--play", action='store_true', default=False, help="Play snake game")

    args = parser.parse_args()
    verbose = args.verbose

    if args.train:
        if verbose:
            print("Start to train AI")

        # TODO Load lastest model here and continue training

    if args.retrain:
        if verbose:
            print("Start to re-train AI with state shape: {0}".format(__default_state_shape__))

        # Abondon the previous model and train a new one

        from train_Qai import TrainAI

        trainai = TrainAI(
            state_shape=__default_state_shape__,
            verbose=verbose
        )
        trainai.start()

    if args.info:
        info = """
            {name}: {version}
            Author: {author}
            Info: {info}
        """.format(
            name='AlphaZero',
            version=__version__,
            author=__author__,
            info=__info__
        )
        print(info)

        help_info = """
            --retrain:  Re-Train AI
            --train:    Train AI
            --verbose:  Show the process information
            --play:     Play with AI
        """
        print("Arguments:")
        print(help_info)

    if args.play:
        print("Play snake game. Please close game in terminal after closing window (i.e, Press Ctrl+C).")
        from retrosnake import RetroSnake

        Nx, Ny = __default_board_shape__
        retrosnake = RetroSnake(Nx, Ny)
        retrosnake.start()