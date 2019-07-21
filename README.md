tf_py_snake

The game snake, made with pygame.
Learning tensorflow, get it to play the game.

├── README.md         # This file
├── snake_ai
│   └── snake_ai.py   # AI version of game, has computer guess which way to go
├── snake_game
│   └── snake.py      # Human playable version of snake

Dependancies:

	python3
	pygame
	tensorflow-2.0.0-beta1
	pandas
	numpy
	sklearn

Human playable (snake_game/snake.py):
	Just hit a direction key and you're off, continue playing until you hit a wall or yourself!
	Once a lose is detected ESCAPE or X button will exit, SPACE will start new game

AI :
	Used a few tutorials to get the basic idea :
		https://github.com/AdnanZahid/SnakeGame
		https://towardsdatascience.com/today-im-going-to-talk-about-a-small-practical-example-of-using-neural-networks-training-one-to-6b2cbd6efdb3
	No code was directly copied, more just the process they used. The AI has two models, one that just trys to survive and not worry about food and the full game AI that will try to close the gap from it to the food.
