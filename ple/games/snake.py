import pygame
import sys
import math

from . import base

from pygame.constants import K_w, K_a, K_s, K_d
from .utils.vec2d import vec2d
from .utils import percent_round_int


class Food(pygame.sprite.Sprite):

    def __init__(self, pos_init, width, color,
                 SCREEN_WIDTH, SCREEN_HEIGHT, rng):
        pygame.sprite.Sprite.__init__(self)

        self.pos = vec2d(pos_init)
        self.color = color

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.width = width
        self.rng = rng

        image = pygame.Surface((width, width))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))
        pygame.draw.rect(
            image,
            color,
            (0, 0, self.width, self.width),
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def new_position(self, snake):
        new_pos = snake.body[0].pos
        snake_body = [s.pos for s in snake.body]
        #snake_body_p = [(s.pos.x,s.pos.y) for s in snake.body]
        #print(snake_body_p)

        while (new_pos in snake_body):
            _x = self.rng.choice(range(
                0, self.SCREEN_WIDTH- self.width, self.width
            ))

            _y = self.rng.choice(range(
                0, self.SCREEN_HEIGHT - self.width, self.width
            ))

            new_pos = vec2d((_x, _y))

        self.pos = new_pos
        #print(new_pos.x,new_pos.y)
        self.rect.center = (self.pos.x, self.pos.y)

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)


class SnakeSegment(pygame.sprite.Sprite):

    def __init__(self, pos_init, width, height, color):
        pygame.sprite.Sprite.__init__(self)

        self.pos = vec2d(pos_init)
        self.color = color
        self.width = width
        self.height = height

        image = pygame.Surface((width, height))
        image.fill((0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.rect(
            image,
            color,
            (0, 0, self.width, self.height),
            0
        )

        self.image = image
        # use half the size
        self.rect = pygame.Rect(pos_init, (self.width / 2, self.height / 2))
        self.rect.center = pos_init

    def draw(self, screen): screen.blit(self.image, self.rect.center)


# basically just holds onto all of them
class SnakePlayer():

    def __init__(self, length, pos_init, width,
                 color, SCREEN_WIDTH, SCREEN_HEIGHT):
        self.dir = vec2d((1, 0))
        self.pos = vec2d(pos_init)
        self.color = color
        self.width = width
        self.length = length
        self.body = []

        # build our body up
        for i in range(self.length):
            self.body.append(
                # makes a neat "zapping" in effect
                SnakeSegment(
                    (self.pos.x - (width) * i, self.pos.y),
                    self.width,
                    self.width,
                    (255,255,0) if i==0 else self.color 
                )
            )
        # we dont add the first few because it cause never actually hit it
        self.body_group = pygame.sprite.Group()
        self.head = self.body[0]

    def update(self, dt, hit=False):
        # Save the last position before move snake body
        last = self.body[-1].pos
        # Shift snake body
        for i in range(self.length - 1, 0, -1):
            scale = 0

            self.body[i].pos.x = self.body[i-1].pos.x
            self.body[i].pos.y = self.body[i-1].pos.y
            #print(i, self.body[i].pos.x, self.body[i].pos.y)
            self.body[i].rect.center = (self.body[i].pos.x, self.body[i].pos.y)

        self.head.pos.x += self.dir.x * self.width
        self.head.pos.y += self.dir.y * self.width
        #print("head", self.body[0].pos.x, self.body[0].pos.y)
        self.head.rect.center = (self.head.pos.x, self.head.pos.y)

        if hit:
            self.length += 1
            #print("hit, len =", self.length)

            # Fancy body color.
            add = 100 if self.length % 2 == 0 else -100
            color = (self.color[0] + add, self.color[1], self.color[2] + add)

            self.body.append(
                SnakeSegment(
                    (last.x, last.y),  # initially off screen?
                    self.width,
                    self.width,
                    color
                )
            )
            if self.length > 3:  # we cant actually hit another segment until this point.
                self.body_group.add(self.body[-1])


    def draw(self, screen):
        for b in self.body[::-1]:
            b.draw(screen)


class Snake(base.PyGameWrapper):
    """
    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    init_length : int (default: 3)
        The starting number of segments the snake has. Do not set below 3 segments. Has issues with hitbox detection with the body for lower values.

    """

    def __init__(self,
                 width=64,
                 height=64,
                 init_length=1):

        actions = {
            "up": K_w,
            "left": K_a,
            "down": K_s,
            "right": K_d
        }

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

        self.player_width = percent_round_int.percent_round_int(width, 0.1)
        self.food_width = percent_round_int.percent_round_int(width, 0.1)
        self.player_color = (0, 255, 0)
        self.food_color = (255, 0, 0)

        self.INIT_POS = (self.player_width*2, self.player_width*2)
        self.init_length = init_length

        self.BG_COLOR = (0, 0, 0)

        self.rewards = {
            "positive": 1.0,
            "negative": -1.0,
            "tick": 0,
            "loss": -5.0,
            "win": 5.0
        }


        # To allow more chance in keydown processing. It seems w/o this snake 
        # cannot make consecutive 1-step turns.
        # Looks like a hack. Not sure whether pygame has a better way to impl 
        # this.
        self.keydown_counter = 0

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                #left = -1
                #right = 1
                #up = -1
                #down = 1

                if key == self.actions["left"] and self.player.dir.x != 1:
                    self.player.dir = vec2d((-1, 0))

                if key == self.actions["right"] and self.player.dir.x != -1:
                    self.player.dir = vec2d((1, 0))

                if key == self.actions["up"] and self.player.dir.y != 1:
                    self.player.dir = vec2d((0, -1))

                if key == self.actions["down"] and self.player.dir.y != -1:
                    self.player.dir = vec2d((0, 1))

    def getActions(self):
        """
        Override the pygamewrapper getActions() func. 
        Output valid actions in fixed order.

        """
        return [self.actions["left"],self.actions["right"],self.actions["up"],self.actions["down"]]

    def getGameState(self):
        """

        Returns
        -------

        dict
            * snake head x position.
            * snake head y position.
            * food x position.
            * food y position.
            * distance from head to each snake segment.

            See code for structure.

        """

        state = {
            "snake_head_x": self.player.head.pos.x,
            "snake_head_y": self.player.head.pos.y,
            "food_x": self.food.pos.x,
            "food_y": self.food.pos.y,
            "snake_body": []
        }

        for s in self.player.body:
            dist = math.sqrt((self.player.head.pos.x - s.pos.x)
                             ** 2 + (self.player.head.pos.y - s.pos.y)**2)
            state["snake_body"].append(dist)

        return state

    def getScore(self):
        return self.score

    def game_over(self):
        return self.lives == -1

    def init(self):
        """
            Starts/Resets the game to its inital state
        """

        self.player = SnakePlayer(
            self.init_length,
            self.INIT_POS,
            self.player_width,
            self.player_color,
            self.width,
            self.height
        )

        self.food = Food((0, 0),
                         self.food_width,
                         self.food_color,
                         self.width,
                         self.height,
                         self.rng
                         )

        self.food.new_position(self.player)

        self.score = 0
        self.ticks = 0
        self.lives = 1

    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        self.keydown_counter += dt

        self.ticks += 1
        self.screen.fill(self.BG_COLOR)
        self._handle_player_events()
        self.score += self.rewards["tick"]

        if self.keydown_counter > 100:
            self.keydown_counter = 0

            hit = pygame.sprite.collide_rect(self.player.head, self.food)
            # Update postions of snake body. Should be done before 
            # self.food.new_position() to avoid new food at the pos
            # of newly inserted snake body.
            self.player.update(dt,hit)
            if hit:  # it hit
                self.score += self.rewards["positive"]
                self.food.new_position(self.player)

            # Crash: head runs into snake body.
            crash = pygame.sprite.spritecollide(
                self.player.head, self.player.body_group, False)
            if len(crash) > 0:
                self.lives = -1

            x_check = (
                self.player.head.pos.x < 0) or (
                self.player.head.pos.x +
                self.player_width /
                2 > self.width)
            y_check = (
                self.player.head.pos.y < 0) or (
                self.player.head.pos.y +
                self.player_width /
                2 > self.height)

            if x_check or y_check:
                self.lives = -1

            if self.lives <= 0.0:
                self.score += self.rewards["loss"]

        self.player.draw(self.screen)
        self.food.draw(self.screen)


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = Snake(width=128, height=128)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        if game.game_over():
            game.init()

        dt = game.clock.tick_busy_loop(30)
        game.step(dt)
        pygame.display.update()
