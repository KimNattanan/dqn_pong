import numpy as np
import torch
import torch.optim.adam
import torch.optim.adam
import time

from game_bot import Game
from network import Network


WINDOW_W, WINDOW_H = 800, 600

game = Game(True)

with torch.no_grad():
  while True:
    game.reset()
    game.upd()
    for i in [1]:
      print("{}..".format(i))
      time.sleep(1)
    while True:
      time.sleep(0.01)
      game.upd()
      if game.isGameOver():
        if game.getReward(0)>0:
          print('plr1 won.')
          game.scores[0] += 1
        else:
          print('plr2 won.')
          game.scores[1] += 1
        break