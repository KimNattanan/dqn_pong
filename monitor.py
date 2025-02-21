import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import deque

class Monitor:
  def __init__(self, n_state, n_action):
    self.rewards_per_ep = deque()
    self.cur_rewards = 0
    self.Qmx_per_ep = deque()
    self.cur_Qmx = deque()

  def push(self, reward, Q):
    self.cur_rewards += reward
    self.cur_Qmx.append(np.max(Q))

  def flush(self, ep):
    self.rewards_per_ep.append([ep, self.cur_rewards])
    self.cur_rewards = 0
    if len(self.cur_Qmx): Qmx_mean = np.mean(self.cur_Qmx)
    else: Qmx_mean = 0
    self.Qmx_per_ep.append([ep, Qmx_mean])
    self.cur_Qmx.clear()
  
  def save(self, name):
    pickle.dump(self.rewards_per_ep,open('./graphs/{}_rewards.pkl'.format(name),'wb'))
    pickle.dump(self.Qmx_per_ep,open('./graphs/{}_Qmx.pkl'.format(name),'wb'))

  def load(self,name):
    self.rewards_per_ep = pickle.load(open('./graphs/{}_rewards.pkl'.format(name),'rb'))
    self.Qmx_per_ep = pickle.load(open('./graphs/{}_Qmx.pkl'.format(name),'rb'))
  
  def plot(self, figsize=(12,10), rewards_avg_window_size=100):
    plt.clf()
    fig, axs = plt.subplots(1,2, figsize=(figsize))
    rewards_per_ep = np.array(self.rewards_per_ep)
    Qmx_per_ep = np.array(self.Qmx_per_ep)

    axs[0].plot(rewards_per_ep[:,0],rewards_per_ep[:,1], label="Episode Reward", linestyle="dotted", color='c')
    axs[0].plot(np.convolve(rewards_per_ep[:,0], np.ones(rewards_avg_window_size)/rewards_avg_window_size, mode="valid"),
                np.convolve(rewards_per_ep[:,1], np.ones(rewards_avg_window_size)/rewards_avg_window_size, mode="valid"),
                label="Moving Average", color='r')
    axs[0].set_title("Episode Reward")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].legend()

    axs[1].plot(Qmx_per_ep[:,0],Qmx_per_ep[:,1], color='c')
    axs[1].set_title("Average Max Q")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Q Value")

    plt.show()