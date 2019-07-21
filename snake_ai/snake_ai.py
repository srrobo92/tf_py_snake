'''#############################################################
# Author : Steven Robichaud
# Date : 7/14/19
#
# Title : skane_ai.py
#
# Description : Learing how to use tensorflow and geting snake to
# collect food on it own and survive as long as possible.
#
#
'''
import sys
import pygame
import random

import numpy as np
import math
from math import pi, asin, sqrt, degrees, radians
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


######################################################################
# Global variables
######################################################################
start_size = 10  # How large to start snake
rando = False  # If to use model(false) or rand data to gen train data
file_name = 'TrainData.csv'  # Name of file to save training data
model_name = 'full_game.h5'  # Name of file that has the tf model
model_input_size = 5  # Full game = 5, survive only = 4
max_deaths = 10000  # Number of deaths for AI to take
clock_speed = 10  # Clock speed
good_moves = 0
bad_moves = 0
debug_file = 'debugging.csv'
######################################################################
# Player Class
# Individual pieces of the snake, handles head movement and posistion
######################################################################


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

#######################################################################
# Food class
# Only used to randomize location of food
#######################################################################


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

#######################################################################
# Game Class
#
# Where most the magic happens, inits pygame, then in game loop preforms
# all the checks if the player has died, collected food, and data as
# the input to the tf model (relative direction, opsticals directly
# around the head of the snake and the angle from the head to the food)
#######################################################################


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
  curDir = False, False, False, False  # Right, Left, Up, Down
  clock = pygame.time.Clock()          # Clock for FPS
  block_size = 10                      # Size of snake in pixels
  food = Food()                        # Food piece
  opsticals = 0, 0, 0, 0               # Right, left, front, if move was beneficial
  ai_input = 0                         # Relative dir AI selected input
  score = 0                            # Score to display
  prediction = []                      # Output of the tf model
  player_died = False                  # Set to true if player has died
  results = 0  # Results after cycle (-1 died, 0 further, 1 closer)
  ang = []     # Angle from head to food
  global start_size
  global file_name


# Constructor
# Init pygame and set screen up
  def __init__(self):
    pygame.init()
    self.screen = pygame.display.set_mode(self.size)

# Run
# Inputs:
# model, The tf model that we are using to predict what to do
# deathCount, Number of times the player has died
# rand, False = use the tf model, True = randomly select inputs and save them to a file
  def run(self, model, deathCount=0, rand=False):
    global bad_moves
    global good_moves
    # Everytime the player dies these variables need to be reset
    # Set restart back to false, set curDir to Right so player starts moving on first frame
    # Clear the previous players body, initialize at least the head, then generate a snake the
    # size of the global variable finally set the AIs select direction to 0 and score back to 0
    self.gameRestart = False
    self.curDir = True, False, False, False
    self.snake = []
    self.snake.append(Player())
    for _ in range(start_size):
      self.snake.append(
          Player(self.snake[-1].x - self.block_size, self.snake[-1].y))
    self.food = []
    self.food = Food()
    self.ai_input = 0
    self.score = 0
    self.player_died = False
    self.ang = 0

    # Main loop, while we aren't exiting the game continue this loop
    while not self.gameExit:

      # Update the game title show the deaths and score
      pygame.display.set_caption(
          'Snake        Deaths : ' + str(deathCount) + '   Score : ' + str(self.score))

      # Check distance and angle to Food
      hyp_bf, self.ang = self.getGoodMoveCheck()

      # Create "user" input
      random.seed()
      # Predict what input to use
      if not(rand):
        prediction = []
        for self.ai_input in range(-1, 2):
          self.opsticals = self.check_opsticals(self.ai_input)
          if model_input_size == 4:
            predict_input = np.ndarray(shape=(1, 4), buffer=np.array(
                [[self.ai_input, self.opsticals[0], self.opsticals[1], self.opsticals[2]]]), dtype=int)
          else:
            predict_input = np.ndarray(shape=(1, 5), buffer=np.array(
                [[self.ai_input, self.opsticals[0], self.opsticals[1], self.opsticals[2], self.ang]]))
          prediction.append(model.predict(predict_input[:]))
        self.ai_input = np.argmax(np.array(prediction)) - 1
      else:
        # Generate Random Input to use
        self.ai_input = random.choice((-1, 0, 1))
        self.opsticals = self.check_opsticals(self.ai_input)

      # Use the selected AI input and the current direction the snake is moving to
      # determine the new direction
      if self.ai_input == -1:
        if self.curDir[0]:
          key_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)
          pygame.event.post(key_event)
        elif self.curDir[1]:
          key_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN)
          pygame.event.post(key_event)
        elif self.curDir[2]:
          key_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
          pygame.event.post(key_event)
        else:
          key_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)
          pygame.event.post(key_event)
      elif self.ai_input == 1:
        if self.curDir[0]:
          key_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN)
          pygame.event.post(key_event)
        elif self.curDir[1]:
          key_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)
          pygame.event.post(key_event)
        elif self.curDir[2]:
          key_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)
          pygame.event.post(key_event)
        else:
          key_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
          pygame.event.post(key_event)

      # Now that new direction is finally decided, update the current direction
      # TODO: Not neccisary right now but can merge this logic with the if else above
      # since currently game will still take human input.
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

      # Check after moving to do before and after comparision
      hyp_af, _ = self.getGoodMoveCheck()

      # Check if food is overlapping
      if self.snake[0].x == self.food.x and self.snake[0].y == self.food.y:
        self.snake.append(Player(self.snake[-1].x, self.snake[-1].y))
        self.food.rand_pos()
        self.score += 1
      # Detect edges of screen
      elif self.snake[0].x < 0 or self.snake[0].x > self.screenWidth - self.block_size or self.snake[0].y < 0 or self.snake[0].y > self.screenHeight - self.block_size or self.check_overlap():
        self.player_died = True

      # print(f"{hyp_af} - {hyp_bf} = {hyp_af - hyp_bf}")
      # print(f"angle = {self.ang}, dir = {self.ai_input}")
      # Check if closer to food
      if not(self.player_died):
        if hyp_af < hyp_bf:
          self.results = 1
          good_moves += 1
        else:
          self.results = 0
          bad_moves += 1
        if rand:
          self.outputToFile(file_name)
      else:
        self.results = -1
        self.outputToFile(debug_file)
        if rand:
          self.outputToFile(file_name)
        return

      # Update screen:
      self.screen.fill(self.black)
      pygame.draw.rect(self.screen, self.red, [
                       self.food.x, self.food.y, self.block_size, self.block_size])
      self.draw_snake()
      pygame.display.update()
      self.clock.tick(clock_speed)

    # We exited the while loop so quit game and close session
    pygame.quit()
    sys.exit()

  # update_pos
  # This function goes through the snake blocks starting from the tail and going to the
  # head updating all their positions to the one infront of it
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

  # draw_snake
  # Draws two blocks for every piece of the snake, one draws the outling and the other fills the body
  def draw_snake(self):
    for block in self.snake:
      pygame.draw.rect(self.screen, self.white, [
          block.x, block.y, self.block_size, self.block_size])
      pygame.draw.rect(self.screen, self.black, [
          block.x, block.y, self.block_size, self.block_size], 1)

  # check_overlap
  # Loops through the body excluding the head of the snake to check if the head is on the same tile
  # as the head
  # Returns True or False
  def check_overlap(self):
    for block in self.snake[1:]:
      if block.x == self.snake[0].x and block.y == self.snake[0].y:
        return True
    return False

  # check_opsticals
  # Inputs:
  # sug_dir, the direction the AI selected to go (relative to current direction)
  # Outputs:
  # rightOps, leftOps, frontOps = 0 if no opstical in direction else 1
  # good_choice = 1 if we moved in a clear direction, 0 if there was an opstical in the direction
  # we moved
  def check_opsticals(self, sug_dir):
    leftOps = 0
    rightOps = 0
    frontOps = 0
    good_choice = 1
    head = self.snake[0]

    if self.curDir[0]:  # Right
      # Check boundaries of window first
      if head.y + self.block_size > self.screenHeight - self.block_size:
        rightOps = 1
      if head.y - self.block_size < 0:
        leftOps = 1
      if head.x + self.block_size > self.screenWidth - self.block_size:
        frontOps = 1
      # Then check if part of our body is right next to us
      for block in self.snake[1:]:
        if block.x == self.snake[0].x and block.y - self.block_size == self.snake[0].y:
          rightOps = 1
        if block.x == self.snake[0].x and block.y + self.block_size == self.snake[0].y:
          leftOps = 1
        if block.x - self.block_size == self.snake[0].x and block.y == self.snake[0].y:
          frontOps = 1
    elif self.curDir[1]:  # Left
      # Check boundaries of window first
      if head.y - self.block_size < 0:
        rightOps = 1
      if head.y + self.block_size > self.screenHeight - self.block_size:
        leftOps = 1
      if head.x - self.block_size < 0:
        frontOps = 1
      # Then check if part of our body is right next to us
      for block in self.snake[1:]:
        if block.x == self.snake[0].x and block.y + self.block_size == self.snake[0].y:
          rightOps = 1
        if block.x == self.snake[0].x and block.y - self.block_size == self.snake[0].y:
          leftOps = 1
        if block.x + self.block_size == self.snake[0].x and block.y == self.snake[0].y:
          frontOps = 1
    elif self.curDir[2]:  # Down
      # Check boundaries of window first
      if head.x - self.block_size < 0:
        rightOps = 1
      if head.x + self.block_size > self.screenWidth - self.block_size:
        leftOps = 1
      if head.y + self.block_size > self.screenHeight - self.block_size:
        frontOps = 1
      # Then check if part of our body is right next to us
      for block in self.snake[1:]:
        if block.x + self.block_size == self.snake[0].x and block.y == self.snake[0].y:
          rightOps = 1
        if block.x - self.block_size == self.snake[0].x and block.y == self.snake[0].y:
          leftOps = 1
        if block.x == self.snake[0].x and block.y - self.block_size == self.snake[0].y:
          frontOps = 1
    elif self.curDir[3]:  # Up
      # Check boundaries of window first
      if head.x + self.block_size > self.screenWidth - self.block_size:
        rightOps = 1
      if head.x - self.block_size < 0:
        leftOps = 1
      if head.y - self.block_size < 0:
        frontOps = 1
      # Then check if part of our body is right next to us
      for block in self.snake[1:]:
        if block.x - self.block_size == self.snake[0].x and block.y == self.snake[0].y:
          rightOps = 1
        if block.x + self.block_size == self.snake[0].x and block.y == self.snake[0].y:
          leftOps = 1
        if block.x == self.snake[0].x and block.y + self.block_size == self.snake[0].y:
          frontOps = 1
    if (rightOps == 1 and sug_dir == 1) or (leftOps == 1 and sug_dir == -1) or (frontOps == 1 and sug_dir == 0):
      good_choice = 0
    return rightOps, leftOps, frontOps, good_choice

  # getOutputForTraining
  # Inputs :
  # ops = opsticals in our direct path left, right and front
  # sug_dir = direction AI has selected to move
  # angle = angle from head of snake to food
  # result = -1 = move ai selected caused death, 0 = moved further from food, 1 = moved closer to food
  def getOutputForTraining(self, ops, sug_dir, angle, result):

    return "\n{},{},{},{},{},{}".format(sug_dir,
                                        ops[0],
                                        ops[1],
                                        ops[2],
                                        angle,
                                        result,
                                        )

  # getOutputForDebug
  def getOutputForDebug(self, sug_dir, angle, result, hyp_af, hyp_bf):

    if (angle < 0 and sug_dir == 1) or (angle > 0 and sug_dir == -1) or (angle == 0 and sug_dir == 0):
      expected = 1
    else:
      expected = 0

    return "\n{},{},{},{},{},{}".format(sug_dir,
                                        angle,
                                        result,
                                        hyp_af,
                                        hyp_bf,
                                        expected
                                        )

  # outputToFile
  # Outputs all data to selected training file
  def outputToFile(self, fn):
    output = self.getOutputForTraining(
        self.opsticals, self.ai_input, self.ang, self.results)
    # print(output)
    file = open(fn, 'a')
    file.write(output)
    file.close()

  # getGoodMoveCheck
  # checks how far food is from us and angle from head of snake to food
  # Outputs :
  # hypotenuse = distance to food
  # angle to food, -1 left of us, 1 right of us
  def getGoodMoveCheck(self):

    head = self.snake[0]

    base = self.food.x - head.x
    perpendicular = self.food.y - head.y

    hypotenuse = math.sqrt(base**2 + perpendicular**2)
    angle = math.atan2(perpendicular, base)

    if self.curDir[1]:  # Left
      if (perpendicular < 0 and base <= 0) or (perpendicular <= 0 and base > 0):
        angle += pi
      else:
        angle -= pi
    elif self.curDir[2]:  # Down
      if base <= 0 and perpendicular < 0:
        angle += (3 / 2) * pi
      else:
        angle -= pi / 2
    elif self.curDir[3]:  # Up
      if base < 0 and perpendicular >= 0:
        angle -= (3 / 2) * pi
      else:
        angle += pi / 2

    return hypotenuse, angle / pi


# "Main" body, load tf model and run game global settings
if __name__ == "__main__":
  deathCount = 0
  model = keras.models.load_model(model_name)
  game = Game()
  for i in range(0, max_deaths):
    game.run(model, deathCount, rando)
    deathCount += 1
  print(f"Good moves : {good_moves}   Bad moves : {bad_moves}")
