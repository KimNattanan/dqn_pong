import numpy as np
import torch
import torch.optim.adam
import torch.optim.adam
import time

from game import Game
from network import Network


WINDOW_W, WINDOW_H = 800, 480
CHECKPOINT_PATH = './checkpoints/duel_left_plr_4400.pt'

game = Game(True)
model = torch.load(CHECKPOINT_PATH,weights_only=False)

model.eval()
with torch.no_grad():
  while True:
    game.reset()
    while True:
      time.sleep(0.0005)
      state = game.getState()
      action = np.argmax(model(torch.tensor(state)).detach().numpy())
      if action==0: game.up(0)
      elif action==1: game.idle(0)
      elif action==2: game.down(0)
      game.upd()
      if game.isGameOver():
        if game.getReward(0)>0:
          print('plr1 won.')
        else:
          print('plr2 won.')
        break