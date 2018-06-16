import numpy as np 
import argparse

__version__ = "0.0.1"
__author__ = "Yang Long"
__info__ = "Play Snake Game with AI"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI")
    parser.add_argument("--retrain", action='store_true', default=False, help="Re-Train AI")
    parser.add_argument("--train",  action='store_true', default=False, help="Train AI")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose")
    parser.add_argument("--info", action='store_true', default=False, help="Show the process information")
    parser.add_argument("--play", action='store_true', default=False, help="Play with AI")

    args = parser.parse_args()
    verbose = args.verbose

    if args.train:
        print("Train AI")

    if args.retrain:
        print("Re-train AI")

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
        print("Play with AI!")