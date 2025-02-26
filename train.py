import pygame  # Ensure correct import of pygame
from main import Game
from config import TRAIN_LOG, SCREEN_WIDTH, SCREEN_HEIGHT
import numpy as np
from utils import load_image
import sys

background_img = load_image("background.png", scale=1)

def train():
    pygame.init()  # Initialize pygame
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))  # Define screen
    pygame.display.set_caption("Q-Learning Space Invaders")
    clock = pygame.time.Clock()  # Create clock object
    game = Game()
    episode_rewards = []
    for episode in range(10000):  # Increase the number of training episodes
        game.reset()
        total_reward = 0
        steps = 0
        max_steps = 100  # Maximum steps per episode
        while steps < max_steps:
            # Handle event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Draw game interface
            screen.blit(background_img, (0, 0))
            game.draw()

            # Get current state
            state = game.get_state()

            # Select action
            action = game.q_learning.get_action(state)

            # Perform action
            reward = game.step(action)
            total_reward += reward

            # Update Q table
            next_state = game.get_state()
            game.q_learning.update_q_table(state, action, reward, next_state)
            game.q_learning.decay_epsilon()

            # Check if game needs to be reset
            if game.game_over or game.win:
                # Simulate space key input to trigger reset
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                game.reset()

            # Limit frame rate
            clock.tick(3000)  # Increase frame rate to 3000 FPS to speed up training

            steps += 1

        episode_rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")

    np.save("q_table.npy", game.q_learning.q_table)

if __name__ == "__main__":
    train()