import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import pickle

from game import Game
from network import Network
from monitor import Monitor


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
  return Q

def train_bot(
  memory_size = 500000,
  reward_discount_factor = 0.99,
  learning_rate = 1e-4,
  number_of_ep = 20000,
  epsilon_decay_factor = 0.9977,
  epsilon_min = 0.05,
  start_epsilon = 0,
  batch_size = 64,
  target_network_update_int = 1000,
  step_per_learn = 10,
  ep_per_save = 100,
  save_name = '',
  plot_name = '',
  checkpoint_path = '',
  plot_load_name = '',
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

  monitor = Monitor(Game.STATE_SHAPE,Game.ACTION_SHAPE)
  if len(plot_load_name):
    monitor.load(plot_load_name)
  
  game = Game(display=display,Rbot=True)
  step = 0
  model[0].train()
  model[1].eval()
  
  for ep in range(begin_ep,number_of_ep+1):
    game.reset()
    state0 = game.getState(0)
    print("game {} start! \t eps: {}".format(ep,epsilon))
    while True:
      action0 = _pred_action(torch.tensor(state0),epsilon,model[0])
      _play_action(game,action0,0)
      game.upd()
      reward0 = game.getReward(0)
      next_state0 = game.getState(0)
      done = game.isGameOver()
      
      model[2].append([state0,action0,reward0,next_state0,done])
      if step%step_per_learn==0 and len(model[2])>=batch_size:
        Q = _fit_batch(model[0],model[1],model[2],optimizer,loss_func,batch_size,reward_discount_factor)
        monitor.push(reward0,Q)
      
      state0 = next_state0
      if step%target_network_update_int==0:
        model[1].load_state_dict(model[0].state_dict())
      step += 1
      if done:
        break

    epsilon = max(epsilon_min,epsilon*epsilon_decay_factor)
    monitor.flush(ep)

    if ep%ep_per_save==0:
      if len(save_name):
        model[0].save('./checkpoints/{}_{}.pt'.format(save_name,ep))
      if len(plot_name):
        monitor.save(plot_name)

  print("score: {} : {}".format(game.scores[0],game.scores[1]))
  del game


def train_ai(
  memory_size = 500000,
  reward_discount_factor = 0.99,
  learning_rate = 1e-4,
  number_of_ep = 20000,
  epsilon_decay_factor = 0.9977,
  epsilon_min = 0.05,
  start_epsilon = 0,
  batch_size = 64,
  target_network_update_int = 1000,
  step_per_learn = 10,
  ep_per_save = 100,
  save_name = '',
  save_name2 = '',
  plot_name = '',
  plot_name2 = '',
  checkpoint_path = '',
  checkpoint_path2 = '',
  plot_load_name = '',
  plot_load_name2 = '',
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

  
  monitor = Monitor(Game.STATE_SHAPE,Game.ACTION_SHAPE)
  monitor2 = Monitor(Game.STATE_SHAPE,Game.ACTION_SHAPE)
  if len(plot_load_name):
    monitor.load(plot_load_name)
  if len(plot_load_name2):
    monitor2.load(plot_load_name2)

  game = Game(display=display)
  step = 0
  model[0].train()
  model[1].eval()
  model2[0].train()
  model2[1].eval()

  for ep in range(begin_ep,number_of_ep+1):
    game.reset()
    state0, state1 = game.getState(0), game.getState(1)
    print("game {} start! \t eps: {}".format(ep,epsilon))
    while True:
      action0, action1 = _pred_action(torch.tensor(state0),epsilon,model[0]), _pred_action(torch.tensor(state1),epsilon,model2[0])
      _play_action(game,action0,0)
      _play_action(game,action1,1)
      game.upd()
      reward0, reward1 = game.getReward(0), game.getReward(1)
      next_state0, next_state1 = game.getState(0), game.getState(1)
      done = game.isGameOver()
      
      model[2].append([state0,action0,reward0,next_state0,done])
      model2[2].append([state1,action1,reward1,next_state1,done])
      if step%step_per_learn==0 and len(model[2])>=batch_size:
        Q = _fit_batch(model[0],model[1],model[2],optimizer,loss_func,batch_size,reward_discount_factor)
        Q2 = _fit_batch(model2[0],model2[1],model2[2],optimizer2,loss_func2,batch_size,reward_discount_factor)
        monitor.push(reward0,Q)
        monitor2.push(reward1,Q2)
      
      state0, state1 = next_state0, next_state1
      if step%target_network_update_int==0:
        model[1].load_state_dict(model[0].state_dict())
        model2[1].load_state_dict(model2[0].state_dict())
      step += 1
      if done:
        break

    epsilon = max(epsilon_min,epsilon*epsilon_decay_factor)
    monitor.flush(ep)
    monitor2.flush(ep)

    if ep%ep_per_save==0:
      if len(save_name):
        model[0].save('./checkpoints/{}_{}.pt'.format(save_name,ep))
      if len(save_name2):
        model2[0].save('./checkpoints/{}_{}.pt'.format(save_name2,ep))
      if len(plot_name):
        monitor.save(plot_name)
      if len(plot_name2):
        monitor2.save(plot_name2)

  print("score: {} : {}".format(game.scores[0],game.scores[1]))
  del game




def train_ai_double(
  memory_size = 500000,
  reward_discount_factor = 0.99,
  learning_rate = 1e-4,
  number_of_ep = 20000,
  epsilon_decay_factor = 0.9977,
  epsilon_min = 0.05,
  start_epsilon = 0,
  batch_size = 64,
  target_network_update_int = 1000,
  step_per_learn = 10,
  ep_per_save = 100,
  save_name = '',
  save_name2 = '',
  plot_name = '',
  plot_name2 = '',
  checkpoint_path = '',
  checkpoint_path2 = '',
  plot_load_name = '',
  plot_load_name2 = '',
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

  monitor = Monitor(Game.STATE_SHAPE,Game.ACTION_SHAPE)
  monitor2 = Monitor(Game.STATE_SHAPE,Game.ACTION_SHAPE)
  if len(plot_load_name):
    monitor.load(plot_load_name)
  if len(plot_load_name2):
    monitor2.load(plot_load_name2)

  game = Game(display=display)
  step = 0
  model[0].train()
  model[1].eval()
  model2[0].train()
  model2[1].eval()

  for ep in range(begin_ep,number_of_ep+1):
    game.reset()
    state0, state1 = game.getState(0), game.getState(1)
    print("game {} start! \t eps: {}".format(ep,epsilon))
    while True:
      action0, action1 = _pred_action(torch.tensor(state0),epsilon,model[0],model2[0]), _pred_action(torch.tensor(state1),epsilon,model2[0],model[0])
      _play_action(game,action0,0)
      _play_action(game,action1,1)
      game.upd()
      reward0, reward1 = game.getReward(0), game.getReward(1)
      next_state0, next_state1 = game.getState(0), game.getState(1)
      done = game.isGameOver()
      
      
      model[2].append([state0,action0,reward0,next_state0,done])
      model2[2].append([state1,action1,reward1,next_state1,done])
      model_to_fit = random.randint(0,1)
      if step%step_per_learn==0 and len(model[2])>=batch_size:
        if model_to_fit==0:
          Q = _fit_batch(model[0],model[1],model[2],optimizer,loss_func,batch_size,reward_discount_factor,model2[1])
          monitor.push(reward0, Q)
        else:
          Q2 = _fit_batch(model2[0],model2[1],model2[2],optimizer2,loss_func2,batch_size,reward_discount_factor,model[1])
          monitor2.push(reward1, Q2)
      
      state0, state1 = next_state0, next_state1
      if step%target_network_update_int==0:
        model[1].load_state_dict(model[0].state_dict())
        model2[1].load_state_dict(model2[0].state_dict())
      step += 1
      if done:
        break

    epsilon = max(epsilon_min,epsilon*epsilon_decay_factor)
    monitor.flush(ep)
    monitor2.flush(ep)

    if ep%ep_per_save==0:
      if len(save_name):
        model[0].save('./checkpoints/{}_{}.pt'.format(save_name,ep))
      if len(save_name2):
        model2[0].save('./checkpoints/{}_{}.pt'.format(save_name2,ep))
      if len(plot_name):
        monitor.save(plot_name)
      if len(plot_name2):
        monitor2.save(plot_name2)

  print("score: {} : {}".format(game.scores[0],game.scores[1]))
  del game

