import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import pickle

from game import Game
from network import Network


WINDOW_W, WINDOW_H = 800, 600

memory_size = 100000
reward_discount_factor = 0.99
learning_rate = 1e-4
number_of_ep = 20000
epsilon = 1.0
epsilon_decay_factor = 0.99999
epsilon_min = 0.01
batch_size = 10
target_network_update_int = 10000
learn_every = 10

game = Game(True)
models = np.array([
  [Network(),Network(),deque(maxlen=memory_size)], # online, target, memo
  [Network(),Network(),deque(maxlen=memory_size)]
],dtype=object)
optimizers = np.array([torch.optim.Adam(models[0,0].parameters(),lr=learning_rate),torch.optim.Adam(models[1,0].parameters(),lr=learning_rate)])
loss_funcs = np.array([nn.MSELoss(),nn.MSELoss()])

cnt = 0
models[0,0].train()
models[0,1].train()
models[1,0].train()
models[1,1].train()
rewards_hist = np.array([]).reshape(0,2)
for ep in range(number_of_ep):
  game.reset()
  state0,state1 = game.getState(0), game.getState(1)
  sum_rewards = np.array([0,0])
  print("game {} start!".format(ep+1))
  while True:
    state_tensor = torch.tensor(state0)
    optimal_action = np.argmax(
      models[0,0](state_tensor).detach().numpy()
      + models[1,0](state_tensor).detach().numpy()
    )
    random_action = random.randint(0,Game.ACTION_SHAPE-1)
    action0 = np.random.choice([random_action,optimal_action],p=[epsilon,1-epsilon])
    if action0==0: game.up(0)
    elif action0==1: game.idle(0)
    elif action0==2: game.down(0)

    state_tensor = torch.tensor(state1)
    optimal_action = np.argmax(
      models[1,0](state_tensor).detach().numpy()
      + models[0,0](state_tensor).detach().numpy()
    )
    random_action = random.randint(0,Game.ACTION_SHAPE-1)
    action1 = np.random.choice([random_action,optimal_action],p=[epsilon,1-epsilon])
    if action1==0: game.up(1)
    elif action1==1: game.idle(1)
    elif action1==2: game.down(1)

    game.upd()

    rewards = np.array([game.getReward(0),game.getReward(1)])
    sum_rewards += rewards

    next_state0 = game.getState(0)
    next_state1 = game.getState(1)
    done = game.isGameOver()
    
    
    models[0,2].append([state0,action0,rewards[0],next_state0,done])
    models[1,2].append([state1,action1,rewards[1],next_state1,done])
    model_to_fit = random.randint(0,1)

    if cnt%learn_every==0 and len(models[model_to_fit,2])>=batch_size:
      batch = np.array(random.sample(models[model_to_fit,2],batch_size),dtype=object)
      state_tensor = torch.tensor(list(batch[:,0]))
      next_state_tensor = torch.tensor(list(batch[:,3]))

      cur_pred = models[model_to_fit,1](next_state_tensor).detach().numpy()
      another_pred = models[1^model_to_fit,1](next_state_tensor).detach().numpy()

      target = batch[:,2] + reward_discount_factor*(another_pred[np.arange(cur_pred.shape[0]),np.argmax(cur_pred,axis=1)])*(1-batch[:,4])
      current_Q = models[model_to_fit,1](state_tensor).detach().numpy()
      current_Q[np.arange(batch_size),list(batch[:,1])] = target
      models[model_to_fit,0].fit(state_tensor,torch.tensor(current_Q),optimizers[0],loss_funcs[0])
    
    state0 = next_state0
    state1 = next_state1
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
    else:
      if rewards[0]>0:
        print('L: hit')
      if rewards[1]>0:
        print('R: hit')
  rewards_hist = np.append(rewards_hist,[sum_rewards],axis=0)

  if (ep+1)%100==0:
    models[0,0].save('./checkpoints/L_online_duel_double_{}.pt'.format(ep+1))
    models[1,0].save('./checkpoints/R_online_duel_double_{}.pt'.format(ep+1))
    pickle.dump(rewards_hist,open('./graphs/rewards_hist.pkl','wb'))
    print(epsilon)