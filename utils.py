# utils.py
import pygame
import os

def load_image(name, scale=1):
    """Load image resource and scale it proportionally"""
    path = os.path.join("assets", "images", name)
    image = pygame.image.load(path).convert_alpha()
    width = int(image.get_width() * scale)
    height = int(image.get_height() * scale)
    return pygame.transform.scale(image, (width, height))

def load_font(name, size):
    """Load font resource"""
    path = os.path.join("assets", "fonts", name)
    return pygame.font.Font(path, size)

def draw_text(screen, text, font, x, y, color=(255, 255, 255)):
    """Draw text on the screen"""
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))