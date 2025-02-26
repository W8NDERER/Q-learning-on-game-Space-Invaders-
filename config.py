# config.py
import os

# Game parameters
SCREEN_WIDTH = 1280  # Adjust screen width
SCREEN_HEIGHT = 720  # Adjust screen height
PLAYER_SPEED = 5
ENEMY_SPEED = 2
BULLET_SPEED = 7
MAX_BULLETS = 5

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Log and save paths
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
TRAIN_LOG = os.path.join(LOG_DIR, "train.log")
TEST_LOG = os.path.join(LOG_DIR, "test.log")
Q_TABLE_SAVE = os.path.join(LOG_DIR, "q_table.npy")