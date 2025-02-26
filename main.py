import pygame
import random
import numpy as np
from config import SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_SPEED, ENEMY_SPEED, BULLET_SPEED, MAX_BULLETS
from qlearn import QLearning
from utils import load_image, load_font, draw_text

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Q-Learning Space Invaders")
clock = pygame.time.Clock()

# Load resources
background_img = load_image("background.png", scale=1)  # Do not scale the background image
player_img = load_image("player.png", scale=0.1)  # Scale down by 10 times
enemy_img = load_image("enemy.png", scale=0.1)  # Scale down by 10 times
bullet_img = load_image("bullet.png", scale=0.1)  # Scale down by 10 times
explosion_img = load_image("explosion.png", scale=0.1)  # Scale down by 10 times

pixel_font = load_font("Sigmar-Regular.ttf", 36)

class Player:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT - 100
        self.speed = PLAYER_SPEED
        self.bullets = []
        self.width = player_img.get_width()
        self.height = player_img.get_height()

    def move(self, direction):
        if direction == "left":
            self.x -= self.speed
        elif direction == "right":
            self.x += self.speed
        self.x = max(0, min(self.x, SCREEN_WIDTH - self.width))

    def shoot(self):
        if len(self.bullets) < MAX_BULLETS:
            self.bullets.append(Bullet(self.x + self.width // 2, self.y))

class Enemy:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH - enemy_img.get_width())
        self.y = random.randint(50, 200)
        self.speed = ENEMY_SPEED
        self.width = enemy_img.get_width()
        self.height = enemy_img.get_height()
        self.shoot_timer = 0  # Shooting timer

    def move(self):
        self.x += self.speed
        if self.x <= 0 or self.x >= SCREEN_WIDTH - self.width:
            self.speed *= -1

    def shoot(self, game):
        """Enemy shooting"""
        if self.shoot_timer == 0:
            game.enemy_bullets.append(EnemyBullet(self.x + self.width // 2, self.y))
            self.shoot_timer = 120  # Set timer to shoot again after 1.2 seconds
        else:
            self.shoot_timer -= 1

class EnemyBullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = bullet_img.get_width()
        self.height = bullet_img.get_height()
        self.speed = 5  # Speed for enemy bullets moving downward

    def move(self):
        self.y += self.speed  # Move downward

class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = bullet_img.get_width()
        self.height = bullet_img.get_height()

    def move(self):
        self.y -= BULLET_SPEED

class Game:
    def __init__(self):
        self.player = Player()
        self.enemies = [Enemy() for _ in range(5)]
        self.explosions = []
        self.enemy_bullets = []  # List for enemy bullets
        self.score = 0
        self.q_learning = QLearning(state_size=(10, 10), action_size=3)
        self.game_over = False  # Add game_over flag
        self.win = False  # Add win flag

    def reset(self):
        """Reset game state"""
        self.player = Player()
        self.enemies = [Enemy() for _ in range(5)]
        self.explosions = []
        self.enemy_bullets = []
        self.score = 0
        self.game_over = False
        self.win = False

    def get_state(self):
        """Get current state"""
        player_x = int(self.player.x // (SCREEN_WIDTH / 10))
        if self.enemies:
            enemy_x = int(self.enemies[0].x // (SCREEN_WIDTH / 10))
        else:
            self.enemies = [Enemy() for _ in range(5)]
            enemy_x = int(self.enemies[0].x // (SCREEN_WIDTH / 10))
        return (player_x, enemy_x)

    def is_done(self):
        """Check if the game is over"""
        # If the player is hit by an enemy bullet, the game ends
        for enemy_bullet in self.enemy_bullets:
            if self.check_collision(enemy_bullet, self.player):
                return True
        return False

    def check_win(self):
        """Check if the game is won"""
        if not self.enemies:
            return True
        return False

    def step(self, action):
        """Perform an action and return the reward"""
        if self.game_over or self.win:
            return 0

        if action == 0:
            self.player.move("left")
        elif action == 1:
            self.player.move("right")
        elif action == 2:
            self.player.shoot()

        self.update()
        reward = self.calculate_reward()
        return reward

    def run(self):
        running = True
        while running:
            screen.blit(background_img, (0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not self.game_over and not self.win:  # If the game is not over and not won
                state = self.get_state()
                action = self.q_learning.get_action(state)

                if action == 0:
                    self.player.move("left")
                elif action == 1:
                    self.player.move("right")
                elif action == 2:
                    self.player.shoot()

                self.update()
                reward = self.calculate_reward()
                next_state = self.get_state()
                self.q_learning.update_q_table(state, action, reward, next_state)
                self.q_learning.decay_epsilon()

            self.draw()
            clock.tick(30)

    def update(self):
        for enemy in self.enemies:
            enemy.move()
            enemy.shoot(self)  # Call the enemy's shooting method

        for bullet in list(self.player.bullets):  # Use a copy of the list for iteration
            bullet.move()
            if bullet.y < 0:  # Remove bullets that go out of the screen
                if bullet in self.player.bullets:
                    self.player.bullets.remove(bullet)

        for enemy_bullet in list(self.enemy_bullets):  # Use a copy of the list for iteration
            enemy_bullet.move()
            if enemy_bullet.y > SCREEN_HEIGHT:  # Remove enemy bullets that go out of the screen
                if enemy_bullet in self.enemy_bullets:
                    self.enemy_bullets.remove(enemy_bullet)

        # Check for collisions
        for enemy in self.enemies:
            if self.check_collision(enemy, self.player):
                self.score -= 10
                self.enemies.remove(enemy)
                self.explosions.append(Explosion(enemy.x, enemy.y))

        for bullet in list(self.player.bullets):  # Use a copy of the list for iteration
            for enemy in self.enemies:
                if self.check_collision(bullet, enemy):
                    self.score += 10
                    self.enemies.remove(enemy)
                    if bullet in self.player.bullets:
                        self.player.bullets.remove(bullet)
                    self.explosions.append(Explosion(enemy.x, enemy.y))

        for enemy_bullet in list(self.enemy_bullets):  # Use a copy of the list for iteration
            if self.check_collision(enemy_bullet, self.player):
                self.score -= 10
                if enemy_bullet in self.enemy_bullets:
                    self.enemy_bullets.remove(enemy_bullet)
                self.explosions.append(Explosion(self.player.x, self.player.y))
                self.game_over = True  # Set game_over to True

        # Check for win condition
        if self.check_win():
            self.win = True

    def calculate_reward(self):
        """Calculate reward"""
        # Reward +20 for defeating an enemy
        if self.score > 0:
            return 20
        # Penalty -15 for being defeated
        elif self.score < 0:
            return -15
        # Penalty -2 if the player is more than halfway across the screen from the nearest enemy
        else:
            player_x = self.player.x
            enemy_x = self.enemies[0].x if self.enemies else 0
            distance = abs(player_x - enemy_x)
            if distance > SCREEN_WIDTH / 2:
                return -2
            return 0

    def check_collision(self, obj1, obj2):
        """Check for collision"""
        return (obj1.x < obj2.x + obj2.width and
                obj1.x + obj1.width > obj2.x and
                obj1.y < obj2.y + obj2.height and
                obj1.y + obj1.height > obj2.y)

    def draw(self):
        """Draw the game"""
        screen.blit(player_img, (self.player.x, self.player.y))
        for enemy in self.enemies:
            screen.blit(enemy_img, (enemy.x, enemy.y))
        for bullet in self.player.bullets:
            screen.blit(bullet_img, (bullet.x, bullet.y))
        for enemy_bullet in self.enemy_bullets:
            screen.blit(bullet_img, (enemy_bullet.x, enemy_bullet.y))  # Use the same bullet image
        for explosion in self.explosions:
            explosion.draw(screen)

        if self.game_over:
            # Display "GAME OVER" text
            draw_text(screen, "GAME OVER", pixel_font, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50, (255, 0, 0))
            draw_text(screen, "Tap to restart", pixel_font, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2, (255, 255, 255))
            pygame.display.flip()
            # Wait for player to click or press a key
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                        self.reset()
                        waiting = False

        elif self.win:
            # Display "You Win" text
            draw_text(screen, "You Win!", pixel_font, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50, (0, 255, 0))
            draw_text(screen, "Tap to restart", pixel_font, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2, (255, 255, 255))
            pygame.display.flip()
            # Wait for player to click or press a key
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                        self.reset()
                        waiting = False

        draw_text(screen, f"Score: {self.score}", pixel_font, 10, 10)
        pygame.display.flip()

class Explosion:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = explosion_img.get_width()
        self.height = explosion_img.get_height()
        self.visible = True
        self.start_time = pygame.time.get_ticks()  # Get the start time of the explosion

    def draw(self, screen):
        if self.visible:
            screen.blit(explosion_img, (self.x, self.y))
            current_time = pygame.time.get_ticks()
            if (current_time - self.start_time) / 1000 >= 0.5:  # Explosion lasts 0.5 seconds
                self.visible = False

if __name__ == "__main__":
    game = Game()
    game.run()