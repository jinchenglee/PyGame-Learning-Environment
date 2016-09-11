from ple.games import Snake 
import numpy as np
import pygame

pygame.init()
game = Snake(width=128, height=128)
game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
game.clock = pygame.time.Clock()
game.rng = np.random.RandomState(24)
game.init()

while True:
    if game.game_over():
        game.init()

    dt = game.clock.tick_busy_loop(5)
    #print(dt)
    game.step(dt)
    pygame.display.update()
