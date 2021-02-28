import pygame
from game import Game
import pandas as pd

pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Create window
size = (700, 500)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Pong neural network")


game = Game(screen)

game.load(0)

# try:
#     x = pd.read_csv("x.csv", delimiter=',').to_numpy().T
#     y = pd.read_csv("y.csv", delimiter=',').to_numpy().T
#     for i in range(1):
#         game.feed(x, y)
#         game.save(0)
#     game.show_cost()
# except Exception:
#     print("Error loading data and training")


# End loop flag
end = False

# We need clock to update screen
clock = pygame.time.Clock()

# Main loop
while end == False:

    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            end = True

    # Check pressed buttons
    keys = pygame.key.get_pressed()

    game.keys(keys)

    # Game logic
    game.update()

    # Check if the ball is bouncing against any of the 4 walls:
    game.check_bounce()

    game.updateX()
    game.check_collide()

    if game.isBounced():
        xy = game.getXY()
        if xy[0].size > 0:
            with open('x.csv', 'a') as f:
                pd.DataFrame(xy[0].T).to_csv(f, header=False, index=None)
            with open('y.csv', 'a') as f:
                pd.DataFrame(xy[1].T).to_csv(f, header=False, index=None)

    # Drawing screen
    screen.fill(BLACK)
    # Draw middle line
    pygame.draw.line(screen, WHITE, [349, 0], [349, 500], 5)
    game.draw()

    game.draw_score()

    # Update screen
    pygame.display.flip()

    # Set frame to 60 per second
    clock.tick(300)

pygame.quit()
