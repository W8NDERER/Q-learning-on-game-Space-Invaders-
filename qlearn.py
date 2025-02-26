# qlearn.py
import numpy as np
import random
from config import LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_START, EPSILON_END, EPSILON_DECAY

class QLearning:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((*state_size, action_size))
        self.epsilon = EPSILON_START

    def get_action(self, state):
        """Select action using epsilon-greedy strategy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Random exploration
        else:
            return np.argmax(self.q_table[state])  # Select the optimal action greedily

    def update_q_table(self, state, action, reward, next_state):
        """Update Q table"""
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """Decay epsilon value"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)