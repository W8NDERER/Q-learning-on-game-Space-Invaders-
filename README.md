# Q-learning-on-game-Space-Invaders-
This is an imitation of the classic game "Space Invaders." Originally released in 1978, "Space Invaders" is a pioneering arcade 2D game that has left a lasting impact on the gaming industry. 

![SI](https://github.com/user-attachments/assets/1bf3faf2-9d74-4f42-9f38-a584f887bf8b)
![image](https://github.com/user-attachments/assets/415b566d-6f70-4a5b-89fe-6c0396879901)

# Game Mode
- **Player vs Enemies:**
The player controls a spaceship at the bottom of the screen, shooting bullets to defeat waves of alien invaders.
- **Enemies Movement:**
Enemies move horizontally across the screen and shoot bullets downward.
- **Objective:**
Defeat all enemies and avoid being hit by enemy bullets.

**You either destroy all the enemy**

![image](https://github.com/user-attachments/assets/1035ff52-1d3d-4fea-b7e0-f17c505a5b5b)



**Or be destroyed**

![image](https://github.com/user-attachments/assets/ec595bc3-71b8-472b-8664-6e6330ce1704)



# Project Principles
- **Q-Learning Algorithm:**
The agent learns to make optimal decisions (actions) based on the current game state, using a Q-table to store the expected utility of each state-action pair.
- **State Representation:**
The game state is discretized into a 10x10 grid, representing the positions of the player and the nearest enemy.
- **Actions:**
The agent can choose to move left, move right, or shoot.
- **Reward Mechanism:**
```+20```reward for defeating an enemy.
```-15```penalty for being hit by an enemy bullet.
```-2```penalty if the player is too far from the nearest enemy.
# Dependencies
Python 3.6+
Pygame
NumPy
Install the required libraries using pip:
```
pip install pygame numpy
```
# How to Run
Normal Game Play
- **1. Run the main game file to play the game(contrlled by agent):**
```
python main.py
```
- **2. Training the Agent**
Run the training script to train the Q-learning agent. *(The training process involves 10,000 episodes and may take approximately 40 minutes to complete)*:
```
python train.py
```

![image](https://github.com/user-attachments/assets/25a18fa0-0ff3-488e-a634-3a6951e7b891)


After training, the learned Q-table will be saved as q_table.npy in the current directory.
- **3. Testing the Trained Agent**
Run the test script to load the trained Q-table and evaluate the agent's performance:
```
python test.py
```
The agent will use the learned Q-table to make decisions in the game environment.
# Project Structure
```
config.py:          #Game and Q-learning parameters configuration.
utils.py:           #Utility functions for loading images and fonts.
qlearn.py:          #Q-learning algorithm implementation.
main.py:            #Main game loop and logic.
train.py:           #Training script for the Q-learning agent.
test.py:            #Testing script for the trained agent.
```
# Notes
The game window size is 1280x720 pixels.
The Q-learning agent uses an ε-greedy exploration strategy with a learning rate of 0.1 and a discount factor of 0.95.
The training process is accelerated by increasing the frame rate to 3000 FPS.

Feel free to modify the game parameters and Q-learning hyperparameters in the config.py file to experiment with different settings.
