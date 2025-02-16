import numpy as np
import torch
import torch.optim.adam
import torch.optim.adam
import time

from game import Game


WINDOW_W, WINDOW_H = 800, 600

@torch.no_grad
def test_single(checkpoint_path, dt=0.01):
  game = Game(True)
  model = torch.load(checkpoint_path,weights_only=False)
  model.eval()
  while True:
    game.reset()
    game.upd()
    time.sleep(0.5)
    while True:
      time.sleep(dt)
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

@torch.no_grad
def test_double(checkpoint_path0, checkpoint_path1, dt=0.01):
  game = Game(True)
  model0 = torch.load(checkpoint_path0,weights_only=False)
  model1 = torch.load(checkpoint_path1,weights_only=False)
  model0.eval()
  model1.eval()
  while True:
    game.reset()
    game.upd()
    time.sleep(0.5)
    while True:
      time.sleep(dt)
      state = torch.tensor(game.getState(0))
      action = np.argmax(model0(state).detach().numpy()+model1(state).detach().numpy())
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


def test_bot(dt=0.01):
  game = Game(True,Lbot=True)
  while True:
    game.reset()
    game.upd()
    time.sleep(0.5)
    while True:
      time.sleep(dt)
      game.upd()
      if game.isGameOver():
        if game.getReward(0)>0:
          print('plr1 won.')
          game.scores[0] += 1
        else:
          print('plr2 won.')
          game.scores[1] += 1
        break