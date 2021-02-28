import pygame
from random import randint

BLACK = (0, 0, 0)


class Ball(pygame.sprite.Sprite):

    def __init__(self, color, width, height):
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)

        pygame.draw.rect(self.image, color, [0, 0, width, height])

        self.velocity = [randint(4, 8), randint(-8, 8)]
        while self.velocity[0] == 0:
            self.velocity = [randint(4, 8), randint(-8, 8)]

        self.rect = self.image.get_rect()

    def update(self):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]

    def reset(self, left=True):
        if left:
            self.velocity = [randint(4, 8), randint(-8, 8)]
        else:
            self.velocity = [-randint(4, 8), randint(-8, 8)]
        self.rect.x = 345
        self.rect.y = 195

    def bounce(self):
        self.velocity[0] = -self.velocity[0]
        self.velocity[1] = randint(-8,8)

    def draw(self, screen):
        screen.blit(self.image, self.rect)