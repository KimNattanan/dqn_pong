import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import pickle

from game import Game
from network import Network


WINDOW_W, WINDOW_H = 800, 600


def _play_action(game,action,plr):
  if action==0: game.up(plr)
  elif action==1: game.idle(plr)
  elif action==2: game.down(plr)

@torch.no_grad
def _pred_action(state_tensor,eps,model,model2=None):
  if model2 is None:
    optimal_action = np.argmax(
      model(state_tensor).detach().numpy()
    )
  else:
    optimal_action = np.argmax(
      model(state_tensor).detach().numpy()
      +model2(state_tensor).detach().numpy()
    )
  random_action = random.randint(0,Game.ACTION_SHAPE-1)
  return np.random.choice([random_action,optimal_action],p=[eps,1-eps])

def _fit_batch(
  online_net, target_net, memo, optimizer, loss_func,
  batch_size, reward_discount_factor,
  target_net2 = None
):
  with torch.no_grad():
    batch = np.array(random.sample(memo,batch_size),dtype=object)
    state_tensor = torch.tensor(list(batch[:,0]))
    next_state_tensor = torch.tensor(list(batch[:,3]))

    pred = target_net(next_state_tensor).detach().numpy()
    if target_net2 is None:
      target = batch[:,2] + reward_discount_factor*(np.max(pred,axis=1))*(1-batch[:,4])
    else:
      pred2 = target_net2(next_state_tensor).detach().numpy()
      target = batch[:,2] + reward_discount_factor*(pred2[np.arange(batch_size),np.argmax(pred,axis=1)])*(1-batch[:,4])

    Q = target_net(state_tensor).detach().numpy()
    Q[np.arange(batch_size),list(batch[:,1])] = target
  online_net.fit(state_tensor,torch.tensor(Q),optimizer,loss_func)

def train_bot(
  memory_size = 500000,
  reward_discount_factor = 0.99,
  learning_rate = 1e-4,
  number_of_ep = 20000,
  epsilon_decay_factor = 0.9977,
  epsilon_min = 0.1,
  start_epsilon = 0,
  batch_size = 64,
  target_network_update_int = 1000,
  step_per_learn = 10,
  ep_per_save = 100,
  save_name = '',
  plot_name = '',
  checkpoint_path = '',
  begin_ep = 1,
  display = True
):

  if start_epsilon: epsilon = start_epsilon
  else: epsilon = max(epsilon_min,epsilon_decay_factor**(begin_ep-1))
  model = np.array([Network(),Network(),deque(maxlen=memory_size)],dtype=object) # online, target, memo

  if len(checkpoint_path):
    model[0] = torch.load(checkpoint_path,weights_only=False)
    model[1].load_state_dict(model[0].state_dict())

  optimizer = torch.optim.Adam(model[0].parameters(),lr=learning_rate)
  loss_func = nn.MSELoss()

  game = Game(display=display,Rbot=True)
  cnt = 0
  model[0].train()
  model[1].eval()
  rewards_hist = np.array([]).reshape(0,2)
  hitcnt_hist = np.array([]).reshape(0,2)
  for ep in range(begin_ep,number_of_ep+1):
    game.reset()
    state0 = game.getState(0)
    sum_rewards = np.array([0,0])
    hitcnt = np.array([0,0])
    print("game {} start! \t eps: {}".format(ep,epsilon))
    while True:
      action0 = _pred_action(torch.tensor(state0),epsilon,model[0])
      _play_action(game,action0,0)

      game.upd()

      rewards = np.array([game.getReward(0)])
      sum_rewards += rewards

      next_state0 = game.getState(0)
      done = game.isGameOver()
      
      
      model[2].append([state0,action0,rewards[0],next_state0,done])

      if cnt%step_per_learn==0 and len(model[2])>=batch_size:
        _fit_batch(model[0],model[1],model[2],optimizer,loss_func,batch_size,reward_discount_factor)
      
      state0 = next_state0
      if cnt%target_network_update_int==0:
        model[1].load_state_dict(model[0].state_dict())
      
      cnt += 1
      if done:
        if rewards[0]>0:
          print('L: win ',game.scores[0])
          game.scores[0] += 1
        else:
          print('R: win ',game.scores[1])
          game.scores[1] += 1
        break
      elif rewards[0]>0:
          print('L: hit')
          hitcnt += 1

    epsilon = max(epsilon_min,epsilon*epsilon_decay_factor)
    rewards_hist = np.append(rewards_hist,[sum_rewards],axis=0)
    hitcnt_hist = np.append(hitcnt_hist,[hitcnt],axis=0)

    if ep%ep_per_save==0:
      if len(save_name):
        model[0].save('./checkpoints/{}_{}.pt'.format(save_name,ep))
      pickle.dump(rewards_hist,open('./graphs/rewards_hist_{}.pkl'.format(plot_name),'wb'))
      pickle.dump(hitcnt_hist,open('./graphs/hitcnt_hist_{}.pkl'.format(plot_name),'wb'))

  print("score: {} : {}".format(game.scores[0],game.scores[1]))
  del game


def train_ai(
  memory_size = 500000,
  reward_discount_factor = 0.99,
  learning_rate = 1e-4,
  number_of_ep = 20000,
  epsilon_decay_factor = 0.9977,
  epsilon_min = 0.1,
  start_epsilon = 0,
  batch_size = 64,
  target_network_update_int = 1000,
  step_per_learn = 10,
  ep_per_save = 100,
  save_name = '',
  save_name2 = '',
  plot_name = '',
  checkpoint_path = '',
  checkpoint_path2 = '',
  begin_ep = 1,
  display = True
):

  if start_epsilon: epsilon = start_epsilon
  else: epsilon = max(epsilon_min,epsilon_decay_factor**(begin_ep-1))
  model = np.array([Network(),Network(),deque(maxlen=memory_size)],dtype=object) # online, target, memo
  model2 = np.array([Network(),Network(),deque(maxlen=memory_size)],dtype=object) # online, target, memo

  if len(checkpoint_path):
    model[0] = torch.load(checkpoint_path,weights_only=False)
    model[1].load_state_dict(model[0].state_dict())
  if len(checkpoint_path2):
    model2[0] = torch.load(checkpoint_path,weights_only=False)
    model2[1].load_state_dict(model[0].state_dict())

  optimizer,optimizer2 = torch.optim.Adam(model[0].parameters(),lr=learning_rate), torch.optim.Adam(model2[0].parameters(),lr=learning_rate)
  loss_func, loss_func2 = nn.MSELoss(), nn.MSELoss()

  game = Game(display=display)
  cnt = 0
  model[0].train()
  model[1].eval()
  model2[0].train()
  model2[1].eval()
  rewards_hist = np.array([]).reshape(0,2)
  hitcnt_hist = np.array([]).reshape(0,2)
  for ep in range(begin_ep,number_of_ep+1):
    game.reset()
    state0, state1 = game.getState(0), game.getState(1)
    sum_rewards = np.array([0,0])
    hitcnt = np.array([0,0])
    print("game {} start! \t eps: {}".format(ep,epsilon))
    while True:
      action0, action1 = _pred_action(torch.tensor(state0),epsilon,model[0]), _pred_action(torch.tensor(state1),epsilon,model2[0])
      _play_action(game,action0,0)
      _play_action(game,action1,1)

      game.upd()

      rewards = np.array([game.getReward(0),game.getReward(1)])
      sum_rewards += rewards

      next_state0, next_state1 = game.getState(0), game.getState(1)
      done = game.isGameOver()
      
      
      model[2].append([state0,action0,rewards[0],next_state0,done])
      model2[2].append([state1,action1,rewards[1],next_state1,done])

      if cnt%step_per_learn==0 and len(model[2])>=batch_size:
        _fit_batch(model[0],model[1],model[2],optimizer,loss_func,batch_size,reward_discount_factor)
        _fit_batch(model2[0],model2[1],model2[2],optimizer2,loss_func2,batch_size,reward_discount_factor)
      
      state0, state1 = next_state0, next_state1
      if cnt%target_network_update_int==0:
        model[1].load_state_dict(model[0].state_dict())
        model2[1].load_state_dict(model2[0].state_dict())
      
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
        if rewards[1]>0:
          print('R: hit')
          hitcnt[1] += 1

    epsilon = max(epsilon_min,epsilon*epsilon_decay_factor)
    rewards_hist = np.append(rewards_hist,[sum_rewards],axis=0)
    hitcnt_hist = np.append(hitcnt_hist,[hitcnt],axis=0)


    if ep%ep_per_save==0:
      if len(save_name):
        model[0].save('./checkpoints/{}_{}.pt'.format(save_name,ep))
      if len(save_name2):
        model2[0].save('./checkpoints/{}_{}.pt'.format(save_name2,ep))
      pickle.dump(rewards_hist,open('./graphs/rewards_hist_{}.pkl'.format(plot_name),'wb'))
      pickle.dump(hitcnt_hist,open('./graphs/hitcnt_hist_{}.pkl'.format(plot_name),'wb'))

  print("score: {} : {}".format(game.scores[0],game.scores[1]))
  del game




def train_ai_double(
  memory_size = 500000,
  reward_discount_factor = 0.99,
  learning_rate = 1e-4,
  number_of_ep = 20000,
  epsilon_decay_factor = 0.9977,
  epsilon_min = 0.1,
  start_epsilon = 0,
  batch_size = 64,
  target_network_update_int = 1000,
  step_per_learn = 10,
  ep_per_save = 100,
  save_name = '',
  save_name2 = '',
  plot_name = '',
  checkpoint_path = '',
  checkpoint_path2 = '',
  begin_ep = 1,
  display = True
):

  if start_epsilon: epsilon = start_epsilon
  else: epsilon = max(epsilon_min,epsilon_decay_factor**(begin_ep-1))
  model = np.array([Network(),Network(),deque(maxlen=memory_size)],dtype=object) # online, target, memo
  model2 = np.array([Network(),Network(),deque(maxlen=memory_size)],dtype=object) # online, target, memo

  if len(checkpoint_path):
    model[0] = torch.load(checkpoint_path,weights_only=False)
    model[1].load_state_dict(model[0].state_dict())
  if len(checkpoint_path2):
    model2[0] = torch.load(checkpoint_path,weights_only=False)
    model2[1].load_state_dict(model[0].state_dict())

  optimizer,optimizer2 = torch.optim.Adam(model[0].parameters(),lr=learning_rate), torch.optim.Adam(model2[0].parameters(),lr=learning_rate)
  loss_func, loss_func2 = nn.MSELoss(), nn.MSELoss()

  game = Game(display=display)
  cnt = 0
  model[0].train()
  model[1].eval()
  model2[0].train()
  model2[1].eval()
  rewards_hist = np.array([]).reshape(0,2)
  hitcnt_hist = np.array([]).reshape(0,2)
  for ep in range(begin_ep,number_of_ep+1):
    game.reset()
    state0, state1 = game.getState(0), game.getState(1)
    sum_rewards = np.array([0,0])
    hitcnt = np.array([0,0])
    print("game {} start! \t eps: {}".format(ep,epsilon))
    while True:
      action0, action1 = _pred_action(torch.tensor(state0),epsilon,model[0],model2[0]), _pred_action(torch.tensor(state1),epsilon,model2[0],model[0])
      _play_action(game,action0,0)
      _play_action(game,action1,1)

      game.upd()

      rewards = np.array([game.getReward(0),game.getReward(1)])
      sum_rewards += rewards

      next_state0, next_state1 = game.getState(0), game.getState(1)
      done = game.isGameOver()
      
      
      model[2].append([state0,action0,rewards[0],next_state0,done])
      model2[2].append([state1,action1,rewards[1],next_state1,done])
      model_to_fit = random.randint(0,1)

      if cnt%step_per_learn==0 and len(model[2])>=batch_size:
        if model_to_fit==0:
          _fit_batch(model[0],model[1],model[2],optimizer,loss_func,batch_size,reward_discount_factor,model2[1])
        else:
          _fit_batch(model2[0],model2[1],model2[2],optimizer2,loss_func2,batch_size,reward_discount_factor,model[1])
      
      state0, state1 = next_state0, next_state1
      if cnt%target_network_update_int==0:
        model[1].load_state_dict(model[0].state_dict())
        model2[1].load_state_dict(model2[0].state_dict())
      
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
        if rewards[1]>0:
          print('R: hit')
          hitcnt[1] += 1

    epsilon = max(epsilon_min,epsilon*epsilon_decay_factor)
    rewards_hist = np.append(rewards_hist,[sum_rewards],axis=0)
    hitcnt_hist = np.append(hitcnt_hist,[hitcnt],axis=0)


    if ep%ep_per_save==0:
      if len(save_name):
        model[0].save('./checkpoints/{}_{}.pt'.format(save_name,ep))
      if len(save_name2):
        model2[0].save('./checkpoints/{}_{}.pt'.format(save_name2,ep))
      pickle.dump(rewards_hist,open('./graphs/rewards_hist_{}.pkl'.format(plot_name),'wb'))
      pickle.dump(hitcnt_hist,open('./graphs/hitcnt_hist_{}.pkl'.format(plot_name),'wb'))

  print("score: {} : {}".format(game.scores[0],game.scores[1]))
  del game

