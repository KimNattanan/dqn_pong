import numpy as np
import torch
import torch.optim.adam
import torch.optim.adam
import time

from game import Game
from network import Network


WINDOW_W, WINDOW_H = 800, 600
CHECKPOINT_PATH = './checkpoints/online_duel_vsbot_0_6300.pt'
CHECKPOINT_PATH2 = './checkpoints/R_online_duel_double_20000.pt'

game = Game(True)
model = torch.load(CHECKPOINT_PATH,weights_only=False)
model2 = torch.load(CHECKPOINT_PATH2,weights_only=False)

model.eval()
model2.eval()
with torch.no_grad():
  while True:
    game.reset()
    game.upd()
    for i in [1]:
      print("{}..".format(i))
      time.sleep(1)
    while True:
      time.sleep(0.01)
      z = game.getReward(0)
      zz = game.getReward(1)
      if z != 0: print(z,zz)
      state = torch.tensor(game.getState(0))
      action = np.argmax(model(state).detach().numpy())
      if action==0: game.up(0)
      elif action==1: game.idle(0)
      elif action==2: game.down(0)
      game.upd()
      if game.isGameOver():
        if game.getReward(0)>0:
          print('plr1 won.')
          game.scores[0] += 1
        else:
          print('plr2 won.')
          game.scores[1] += 1
        break