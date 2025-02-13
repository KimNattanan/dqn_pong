import random
import numpy as np
from collections import deque

from game import Game
from network import Network


WINDOW_W, WINDOW_H = 800, 480

memory_size = 100000
reward_discount_factor = 0.99
number_of_ep = 10000
epsilon = 1.0
epsilon_decay_factor = 0.999
epsilon_min = 0.01
batch_size = 32
target_network_update_int = 500

game = Game(True,0.05)
networks = [(Network(),Network()),(Network(),Network())]
memos = [deque(maxlen=memory_size),deque(maxlen=memory_size)]

cnt = 0
networks[0][0].train()
networks[0][1].train()
networks[1][0].train()
networks[1][1].train()
for ep in range(number_of_ep):
  game.reset()
  state = game.getState()
  print("game {} start!".format(ep+1))
  while True:
    for i in [0,1]:
      online_network,target_network = networks[i]
      optimal_action = np.argmax(online_network(state).detach().numpy())
      random_action = random.randint(0,Game.ACTION_SHAPE-1)
      action = np.random.choice([random_action,optimal_action],p=[epsilon,1-epsilon])
      
      if action==0: game.up(i)
      elif action==1: game.idle(i)
      elif action==2: game.down(i)

    game.upd()

    next_state = game.getState()
    done = game.isGameOver()
    if done:
      if game.getReward(0)>0: print('\t\t\tplr1 won')
      else: print('\t\t\tplr2 won')
    
    for i in [0,1]:
      online_network,target_network = networks[i]
      memo = memos[i]
      reward = game.getReward(i)

      memo.append([state,action,reward,next_state,done])
      if len(memo)>=batch_size:
        batch = np.array(random.sample(memo,batch_size),dtype=object)
        target = batch[:,2] + reward_discount_factor*(np.max(target_network(list(batch[:,3])).detach().numpy(),axis=1))*(1-batch[:,4])
        current_Q = target_network(list(batch[:,0])).detach().numpy()
        current_Q[np.arange(batch_size),list(batch[:,1])] = target
        for x,y in zip(batch[:,0],current_Q):
          online_network.fit(x,y)
    
    state = next_state
    if epsilon>epsilon_min: epsilon = epsilon*epsilon_decay_factor
    if cnt%target_network_update_int==0:
      for i in [0,1]:
        online_network,target_network = networks[i]
        target_network.load_state_dict(online_network.state_dict())
    cnt += 1
    if done:
      break

  if (ep+1)%100==0 or ep==0:
    networks[0][0].save('./checkpointstf/test_left_{}.pt'.format(ep+1))
    networks[1][0].save('./checkpointstf/test_right_{}.pt'.format(ep+1))