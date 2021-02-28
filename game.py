import pygame
from paddle import Paddle
from ball import Ball
import numpy as np
from neural import NeuralNetwork

WHITE = (255, 255, 255)
LASTHIT_PADDLE_A = 0
LASTHIT_PADDLE_B = 1


class Game:
    def __init__(self, screen):
        self.screen = screen

        # Create two paddles
        self.paddleA = Paddle(WHITE, 10, 100)
        self.paddleA.rect.x = 20
        self.paddleA.rect.y = 200

        self.paddleB = Paddle(WHITE, 10, 100)
        self.paddleB.rect.x = 670
        self.paddleB.rect.y = 200

        self.ball = Ball(WHITE, 10, 10)
        self.ball.rect.x = 345
        self.ball.rect.y = 195

        # Add all sprites to list
        self.all_sprites_list = pygame.sprite.Group()
        self.all_sprites_list.add(self.paddleA)
        self.all_sprites_list.add(self.paddleB)
        self.all_sprites_list.add(self.ball)

        self.X = np.array([])
        self.Y = np.array([])
        self.lasthit = LASTHIT_PADDLE_A
        self.laststart_left = False

        # self.nnA = NeuralNetwork([4, 32, 4, 4, 1])
        # self.nnB = NeuralNetwork([4, 8, 4, 4, 1])
        # self.nnA = NeuralNetwork([4, 32, 4, 4, 1])
        # self.nnB = NeuralNetwork([4, 32, 8, 1])
        # self.nnA = NeuralNetwork([4, 8, 4, 1])
        # self.nnB = NeuralNetwork([4, 16, 4, 1])
        self.nnA = NeuralNetwork([4, 16, 4, 4, 1])
        self.nnB = NeuralNetwork([4, 16, 8, 1])

    def draw(self):
        self.all_sprites_list.draw(self.screen)

    def update(self):
        self.all_sprites_list.update()

    def keys(self, keys):
        yA = self.nnA.forward_propagation(
                np.array([(self.ball.rect.x + 5) / 700,
                         (self.ball.rect.y + 5) / 500,
                         self.ball.velocity[0] / 8,
                         self.ball.velocity[1] / 8]).reshape(4, 1))
        if (yA - 0.1) / 0.8 * 500 < self.paddleA.rect.y:
            self.paddleA.moveUp(5)
        else:
            self.paddleA.moveDown(5)

        yB = self.nnB.forward_propagation(
                np.array([(700 - self.ball.rect.x + 5) / 700,
                         (self.ball.rect.y + 5) / 500,
                         -self.ball.velocity[0] / 8,
                         self.ball.velocity[1] / 8]).reshape(4, 1))
        if (yB - 0.1) / 0.8 * 500 < self.paddleB.rect.y:
            self.paddleB.moveUp(5)
        else:
            self.paddleB.moveDown(5)

        # paddleBMid = self.paddleB.rect.y + self.paddleB.rect.height / 2
        # if self.ball.rect.y < paddleBMid:
        # # if keys[pygame.K_UP]:
        #     self.paddleB.moveUp(5)
        # if self.ball.rect.y > paddleBMid:
        # # if keys[pygame.K_DOWN]:
        #     self.paddleB.moveDown(5)

    def check_bounce(self):
        if self.ball.rect.x >= 690:
            self.paddleA.score += 1
            self.ball.velocity[0] = -self.ball.velocity[0]
            self.laststart_left = not self.laststart_left
            self.ball.reset(self.laststart_left)
            self.X = np.array([])
            self.Y = np.array([])
            if self.laststart_left:
                self.lasthit = LASTHIT_PADDLE_A
            else:
                self.lasthit = LASTHIT_PADDLE_B
        if self.ball.rect.x <= 0:
            self.paddleB.score += 1
            self.ball.velocity[0] = -self.ball.velocity[0]
            self.laststart_left = not self.laststart_left
            self.ball.reset(self.laststart_left)
            self.X = np.array([])
            self.Y = np.array([])
            if self.laststart_left:
                self.lasthit = LASTHIT_PADDLE_A
            else:
                self.lasthit = LASTHIT_PADDLE_B
        if self.ball.rect.y > 490:
            self.ball.velocity[1] = -self.ball.velocity[1]
        if self.ball.rect.y < 0:
            self.ball.velocity[1] = -self.ball.velocity[1]

    def check_collide(self):
        if pygame.sprite.collide_mask(self.ball, self.paddleA):
            if self.lasthit == LASTHIT_PADDLE_B:
                self.ball.bounce()
                self.Y = np.full((1, int(self.X.size / 4)), np.array([self.paddleA.rect.y / 500 * 0.8 + 0.1]))
                self.lasthit = LASTHIT_PADDLE_A
        elif pygame.sprite.collide_mask(self.ball, self.paddleB):
            if self.lasthit == LASTHIT_PADDLE_A:
                self.ball.bounce()
                self.Y = np.full((1, int(self.X.size / 4)), np.array([self.paddleB.rect.y / 500 * 0.8 + 0.1]))
                self.lasthit = LASTHIT_PADDLE_B

    def draw_score(self):
        font = pygame.font.Font(None, 74)
        text = font.render(str(self.paddleA.score), 1, WHITE)
        self.screen.blit(text, (250,10))
        text = font.render(str(self.paddleB.score), 1, WHITE)
        self.screen.blit(text, (420,10))

    def updateX(self):
        if self.lasthit == LASTHIT_PADDLE_B:
            self.X = np.append(self.X, np.array([(self.ball.rect.x + 5) / 700, (self.ball.rect.y + 5) / 500, self.ball.velocity[0] / 8, self.ball.velocity[1] / 8]))
        else:
            self.X = np.append(self.X, np.array([(700 - self.ball.rect.x + 5) / 700, (self.ball.rect.y + 5) / 500, -self.ball.velocity[0] / 8, self.ball.velocity[1] / 8]))

    def isBounced(self):
        if self.Y.size > 0:
            return True
        return False

    def getXY(self):
        t = (self.X.reshape(4, -1, order='F').copy(), self.Y.reshape(1, -1).copy())
        self.X = np.array([])
        self.Y = np.array([])
        return t

    def feed(self, X, Y):
        self.nnA.feed(X, Y)
        self.nnB.feed(X, Y)

    def save(self, num):
        for i in range(len(self.nnA.w)):
            np.save(f"weights_{num}_{i}_1.npy", self.nnA.w[i])
            np.save(f"bias_{num}_{i}_1.npy", self.nnA.b[i])
        for i in range(len(self.nnB.w)):
            np.save(f"weights_{num}_{i}_2.npy", self.nnB.w[i])
            np.save(f"bias_{num}_{i}_2.npy", self.nnB.b[i])

    def load(self, num):
        try:
            for i in range(len(self.nnA.w)):
                self.nnA.w[i] = np.load(f"weights_{num}_{i}_1.npy")
                self.nnA.b[i] = np.load(f"bias_{num}_{i}_1.npy")
        except Exception:
            print("Error loading parameters A")
        try:
            for i in range(len(self.nnB.w)):
                self.nnB.w[i] = np.load(f"weights_{num}_{i}_2.npy")
                self.nnB.b[i] = np.load(f"bias_{num}_{i}_2.npy")
        except Exception:
            print("Error loading parameters B")

    def show_cost(self):
        self.nnA.show_cost()
        self.nnB.show_cost()




