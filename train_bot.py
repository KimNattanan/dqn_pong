import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import pickle

from game_bot import Game
from network import Network


WINDOW_W, WINDOW_H = 800, 600

memory_size = 500000
reward_discount_factor = 0.99
learning_rate = 1e-4
number_of_ep = 20000
epsilon_decay_factor = 0.995
epsilon_min = 0.1
batch_size = 64
target_network_update_int = 1000
learn_every = 10

begin_ep = 1
model0_path = './checkpoints/online_duel_vsbot_0_{}.pt'.format(begin_ep-1)
epsilon = 1.0

models = np.array([
  [Network(),Network(),deque(maxlen=memory_size)], # online, target, memo
  [Network(),Network(),deque(maxlen=memory_size)]
],dtype=object)

if begin_ep>1:
  epsilon = max(epsilon_min,epsilon_decay_factor**(begin_ep-1))
  models[0,0] = torch.load(model0_path,weights_only=False)
  models[0,1] = torch.load(model0_path,weights_only=False)

optimizers = np.array([torch.optim.Adam(models[0,0].parameters(),lr=learning_rate),torch.optim.Adam(models[1,0].parameters(),lr=learning_rate)])
loss_funcs = np.array([nn.MSELoss(),nn.MSELoss()])

game = Game(True)
cnt = 0
models[0,0].train()
models[0,1].eval()
# models[1,0].train()
# models[1,1].eval()
rewards_hist = np.array([]).reshape(0,2)
hitcnt_hist = np.array([]).reshape(0,2)
for ep in range(begin_ep,number_of_ep+1):
  game.reset()
  state0 = game.getState(0)
  sum_rewards = np.array([0,0])
  hitcnt = np.array([0,0])
  print("game {} start!".format(ep))
  while True:
    state_tensor = torch.tensor(state0)
    with torch.no_grad():
      optimal_action = np.argmax(
        models[0,0](state_tensor).detach().numpy()
        # + models[1,0](state_tensor).detach().numpy()
      )
    random_action = random.randint(0,Game.ACTION_SHAPE-1)
    action0 = np.random.choice([random_action,optimal_action],p=[epsilon,1-epsilon])
    if action0==0: game.up(0)
    elif action0==1: game.idle(0)
    elif action0==2: game.down(0)

    game.upd()

    rewards = np.array([game.getReward(0)])
    sum_rewards += rewards

    next_state0 = game.getState(0)
    done = game.isGameOver()
    
    
    models[0,2].append([state0,action0,rewards[0],next_state0,done])
    # model_to_fit = random.randint(0,1)
    model_to_fit = 0

    if cnt%learn_every==0 and len(models[model_to_fit,2])>=batch_size:
      with torch.no_grad():
        batch = np.array(random.sample(models[model_to_fit,2],batch_size),dtype=object)
        state_tensor = torch.tensor(list(batch[:,0]))
        next_state_tensor = torch.tensor(list(batch[:,3]))

        cur_pred = models[model_to_fit,1](next_state_tensor).detach().numpy()
        # another_pred = models[1^model_to_fit,1](next_state_tensor).detach().numpy()

        target = batch[:,2] + reward_discount_factor*(np.max(cur_pred,axis=1))*(1-batch[:,4])
        current_Q = models[model_to_fit,1](state_tensor).detach().numpy()
        current_Q[np.arange(batch_size),list(batch[:,1])] = target
      models[model_to_fit,0].fit(state_tensor,torch.tensor(current_Q),optimizers[0],loss_funcs[0])
    
    state0 = next_state0
    if cnt%target_network_update_int==0:
      models[0,1].load_state_dict(models[0,0].state_dict())
      # models[1,1].load_state_dict(models[1,0].state_dict())
    cnt += 1

    if done:
      if rewards[0]>0:
        print('L: win ',game.scores[0])
        game.scores[0] += 1
      else:
        print('R: win ',game.scores[1])
        game.scores[1] += 1
      break
    else:
      if rewards[0]>0:
        print('L: hit')
        hitcnt[0] += 1
      # elif rewards[1]>0:
      #   print('R: hit')
      #   hitcnt[1] += 1

  epsilon = max(epsilon_min,epsilon*epsilon_decay_factor)
  rewards_hist = np.append(rewards_hist,[sum_rewards],axis=0)
  hitcnt_hist = np.append(hitcnt_hist,[hitcnt],axis=0)

  if ep%100==0:
    models[0,0].save('./checkpoints/online_duel_vsbot_0_{}.pt'.format(ep+1))
    pickle.dump(rewards_hist,open('./graphs/rewards_hist.pkl','wb'))
    pickle.dump(rewards_hist,open('./graphs/hitcnt_hist.pkl','wb'))
    print(epsilon)