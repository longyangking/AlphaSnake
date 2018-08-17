# AlphaSnake
> Retro Snake Game with Artificial Intelligence

![](./pics/retrosnake.png)

## Features
+ Reinforcement Learning
+ Monte Carlo Tree Search

## Dependencies
+ Python 3+
+ Keras, Tensorflow
+ PyQt5 (For UI)

## How to use
+ Play Retro Snake by yourself (Linux)  

        cd alphasnake
        python alphasnake.py --play

+ Re-Train AI model (recommand to do this in GPU server)

        cd alphasnake.py
        python alphasnake.py --retrain --verbose

+ Continue to train previous AI model (recommand to do this in GPU server)

        cd alphasnake.py
        python alphasnake.py --train --verbose

+ Play Snake game by AI

        cd alphasnake.py
        python alphasnake.py --playai

## Information
You can get the similar theoretical details in the paper of DeepMind company about AlphaZero [1,2].

## References
[1]Silver, David, et al. "Mastering the game of Go without human knowledge." Nature 550.7676 (2017): 354.  
[2]Silver, David, et al. "Mastering the game of Go with deep neural networks and tree search." nature 529.7587 (2016): 484.  