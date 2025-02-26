from main import Game
from config import TEST_LOG
import numpy as np

def test():
    game = Game()
    game.q_learning.q_table = np.load("q_table.npy") # Load trained Q table
    while True:
        game.run()

if __name__ == "__main__":
    test()