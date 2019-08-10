import sys
import pygame
import random

start_size = 5  # How large to start snake

# Player Class
# Handles how big the player is, the direction the player is moving


class Player:
  speed = 10

  def __init__(self, x=200, y=200):
    self.x = x
    self.y = y

  def moveRight(self):
    self.x = self.x + self.speed

  def moveLeft(self):
    self.x = self.x - self.speed

  def moveUp(self):
    self.y = self.y - self.speed

  def moveDown(self):
    self.y = self.y + self.speed

  def reset(self):
    self.x = 200
    self.y = 200


class Food:
  x = 0
  y = 0
  size = 10

  def __init__(self):
    self.rand_pos()

  def rand_pos(self):
    random.seed()
    self.x = random.randrange(0, 400 - self.size, self.size)
    self.y = random.randrange(0, 400 - self.size, self.size)


class Game:
  screenWidth = 400                    # Screen Width
  screenHeight = 400                   # Screen Height
  size = screenWidth, screenHeight     # Array of the screen size
  black = 0, 0, 0                      # The color settings for black
  white = 255, 255, 255                # The color settings for white
  red = 255, 0, 0                      # The color settings for red
  snake = []                           # Instance of the player class
  gameExit = False                     # Game state, Exit/Quit state
  gamePause = False                    # Game state, Pause state
  gameRestart = False                  # Game state, Restart state
  curDir = True, False, False, False  # Right, Left, Up, Down
  clock = pygame.time.Clock()          # Clock for FPS
  fps = 10                             # FPS
  score = 0                            # Score to display
  block_size = 10                      # Size of snake in pixels
  food = Food()                        # Food piece
  global start_size

  def __init__(self):
    pygame.init()
    self.screen = pygame.display.set_mode(self.size)
    self.snake.append(Player())
    for _ in range(start_size):
      self.snake.append(
          Player(self.snake[-1].x - self.block_size, self.snake[-1].y))

  def run(self):

    self.score = 0
    self.curDir = True, False, False, False

    while not self.gameExit:

      pygame.display.set_caption('Snake' + '        Score:' + str(self.score))

      # Get user input
      for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
          self.gameExit = True
        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_RIGHT and not(self.curDir[1]):
            self.curDir = True, False, False, False
          elif event.key == pygame.K_LEFT and not(self.curDir[0]):
            self.curDir = False, True, False, False
          elif event.key == pygame.K_DOWN and not(self.curDir[3]):
            self.curDir = False, False, True, False
          elif event.key == pygame.K_UP and not(self.curDir[2]):
            self.curDir = False, False, False, True

      # Update snake pos
      self.update_pos()

      # Check if food is overlapping
      if self.snake[0].x == self.food.x and self.snake[0].y == self.food.y:
        self.snake.append(Player(self.snake[-1].x, self.snake[-1].y))
        self.food.rand_pos()
        self.score += 1
      # Detect edges of screen
      elif self.snake[0].x < 0 or self.snake[0].x > self.screenWidth - self.block_size or self.snake[0].y < 0 or self.snake[0].y > self.screenHeight - self.block_size or self.check_overlap():
        self.restart()

        # Update screen
      self.screen.fill(self.black)
      pygame.draw.rect(self.screen, self.red, [
                       self.food.x, self.food.y, self.block_size, self.block_size])
      self.draw_snake()
      pygame.display.update()
      self.clock.tick(self.fps)

    pygame.quit()
    sys.exit()

  def restart(self):
    while not self.gameRestart and not self.gameExit:
      # Get user input
      for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
          self.gameExit = True
        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_SPACE:
            self.gameRestart = True
    if self.gameRestart:
      self.gameRestart = False
      self.curDir = True, False, False, False
      self.snake = []
      self.snake.append(Player())
      for _ in range(start_size):
        self.snake.append(
            Player(self.snake[-1].x - self.block_size, self.snake[-1].y))
      self.run()
    else:
      pygame.quit()
      sys.exit()

  def update_pos(self):
    for i, block in reversed(list(enumerate(self.snake))):
      if i == 0:
        # Act on user input
        if self.curDir[0]:
          block.moveRight()
        elif self.curDir[1]:
          block.moveLeft()
        elif self.curDir[2]:
          block.moveDown()
        elif self.curDir[3]:
          block.moveUp()
      else:
        block.x = self.snake[i - 1].x
        block.y = self.snake[i - 1].y

  def draw_snake(self):
    for block in self.snake:
      pygame.draw.rect(self.screen, self.white, [
          block.x, block.y, self.block_size, self.block_size])
      pygame.draw.rect(self.screen, self.black, [
          block.x, block.y, self.block_size, self.block_size], 1)

  def check_overlap(self):
    for block in self.snake[1:]:
      if block.x == self.snake[0].x and block.y == self.snake[0].y:
        return True
    return False


if __name__ == "__main__":
  snake_game = Game()
  snake_game.run()


def main():
  pass
