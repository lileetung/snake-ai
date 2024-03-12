# Snake AI

## Overview

This project utilizes a Deep Q-Network (DQN) algorithm to autonomously play the classic game of Snake. The goal is to train a reinforcement learning agent that can effectively navigate the game environment, avoid obstacles, and maximize its score by eating apples. The project demonstrates the application of convolutional neural networks (CNNs) within a reinforcement learning framework to process spatial data and make decisions.

![snakeAI](https://github.com/lileetung/snake-ai/assets/83776772/1bbe8957-4f28-44c0-84e6-7cc2c4734ccd)

## Technical Highlights

- **Reinforcement Learning**: Implementing a DQN to handle the decision-making process of the snake, learning from its environment through rewards and penalties.
- **Environment Design**: Crafting a game environment that accurately represents the Snake game, including the movement of the snake, collision detection, and apple placement.
- **CNN Integration**: Utilizing convolutional neural networks to interpret the game state as input and decide on the best action to take next.
- **Reward System**: Designing a reward system that encourages the snake to eat apples while avoiding collisions with itself or the game boundaries.

## Prerequisites

Before you can run the project, ensure you have Python 3.x installed on your machine. You should also have basic familiarity with reinforcement learning concepts and PyTorch.

- Pygame
- NumPy
- PyTorch

You can install the dependencies with pip:

```bash
pip install pygame numpy torch
```

## How to Run

1. Clone the repository to your local machine.
2. Navigate to the cloned directory.
3. Run the `train.py` script to train the model:
    ```bash
    python train.py
    ```
4. After training, you can run the `test.py` script to see the trained model play the game:
    ```bash
    python test.py
    ```

Note: The training process can take a considerable amount of time, depending on your hardware capabilities.
