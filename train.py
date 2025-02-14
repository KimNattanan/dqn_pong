import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import pickle

from game import Game
from network import Network


WINDOW_W, WINDOW_H = 800, 480

memory_size = 100000
reward_discount_factor = 0.99
number_of_ep = 10000
epsilon = 1.0
epsilon_decay_factor = 0.999
epsilon_min = 0.01
batch_size = 10
target_network_update_int = 500

game = Game(True)
models = np.array([
  [Network(),Network(),deque(maxlen=memory_size)],
  [Network(),Network(),deque(maxlen=memory_size)]
],dtype=object)
optimizers = np.array([torch.optim.Adam(models[0,0].parameters(),lr=0.001),torch.optim.Adam(models[1,0].parameters(),lr=0.001)])
loss_funcs = np.array([nn.MSELoss(),nn.MSELoss()])

cnt = 0
models[0,0].train()
models[0,1].train()
models[1,0].train()
models[1,1].train()
rewards_hist = np.array([]).reshape(0,2)
for ep in range(number_of_ep):
  game.reset()
  state = game.getState()
  sum_rewards = np.array([0,0])
  print("game {} start!".format(ep+1))
  while True:
    optimal_action = np.argmax(models[0,0](torch.tensor(state)).detach().numpy())
    random_action = random.randint(0,Game.ACTION_SHAPE-1)
    action0 = np.random.choice([random_action,optimal_action],p=[epsilon,1-epsilon])
    if action0==0: game.up(0)
    elif action0==1: game.idle(0)
    elif action0==2: game.down(0)

    optimal_action = np.argmax(models[1,0](torch.tensor(state)).detach().numpy())
    random_action = random.randint(0,Game.ACTION_SHAPE-1)
    action1 = np.random.choice([random_action,optimal_action],p=[epsilon,1-epsilon])
    if action1==0: game.up(1)
    elif action1==1: game.idle(1)
    elif action1==2: game.down(1)

    game.upd()

    rewards = np.array([game.getReward(0),game.getReward(1)])
    sum_rewards += rewards

    if rewards[0]==1:
      print('L: hit')
    if rewards[1]==1:
      print('R: hit')

    next_state = game.getState()
    done = game.isGameOver()
    
    
    models[0,2].append([state,action0,rewards[0],next_state,done])
    if len(models[0,2])>=batch_size:
      batch = np.array(random.sample(models[0,2],batch_size),dtype=object)
      target = batch[:,2] + reward_discount_factor*(np.max(models[0,1](torch.tensor(list(batch[:,3]))).detach().numpy(),axis=1))*(1-batch[:,4])
      current_Q = models[0,1](torch.tensor(list(batch[:,0]))).detach().numpy()
      current_Q[np.arange(batch_size),list(batch[:,1])] = target
      x_batch = torch.tensor(batch[:,0].tolist(),dtype=torch.float32)
      y_batch = torch.tensor(current_Q,dtype=torch.float32)
      models[0,0].fit(x_batch,y_batch,optimizers[0],loss_funcs[0])

    models[1,2].append([state,action1,rewards[1],next_state,done])
    if len(models[1,2])>=batch_size:
      batch = np.array(random.sample(models[1,2],batch_size),dtype=object)
      target = batch[:,2] + reward_discount_factor*(np.max(models[1,1](torch.tensor(list(batch[:,3]))).detach().numpy(),axis=1))*(1-batch[:,4])
      current_Q = models[1,1](torch.tensor(list(batch[:,0]))).detach().numpy()
      current_Q[np.arange(batch_size),list(batch[:,1])] = target
      x_batch = torch.tensor(batch[:,0].tolist(),dtype=torch.float32)
      y_batch = torch.tensor(current_Q,dtype=torch.float32)
      models[1,0].fit(x_batch,y_batch,optimizers[1],loss_funcs[1])
    
    state = next_state
    if epsilon>epsilon_min: epsilon = epsilon*epsilon_decay_factor
    if cnt%target_network_update_int==0:
      models[0,1].load_state_dict(models[0,0].state_dict())
      models[1,1].load_state_dict(models[1,0].state_dict())
    cnt += 1

    if done:
      if rewards[0]>0:
        game.scores[0] += 1
        print('L: win ',game.scores[0])
      else:
        game.scores[1] += 1
        print('R: win ',game.scores[1])
      break
  rewards_hist = np.append(rewards_hist,[sum_rewards],axis=0)

  if (ep+1)%100==0:
    models[0,0].save('./checkpoints/duel_L_online_{}.pt'.format(ep+1))
    models[0,1].save('./checkpoints/duel_L_target_{}.pt'.format(ep+1))
    models[1,0].save('./checkpoints/duel_R_online_{}.pt'.format(ep+1))
    models[1,1].save('./checkpoints/duel_R_target_{}.pt'.format(ep+1))
    pickle.dump(rewards_hist,open('./graphs/rewards_hist.pkl','wb'))