import numpy as np
import pickle
import matplotlib.pyplot as plt

class Monitor:
  def __init__(self, n_state, n_action):
    self.rewards_per_ep = np.array([]).reshape(0,2)
    self.cur_rewards = 0
    self.Qmx_per_ep = np.array([]).reshape(0,2)
    self.cur_Qmx = np.array([])

  def push(self, reward, Q):
    self.cur_rewards += reward
    self.cur_Qmx = np.append(self.cur_Qmx, np.max(Q))

  def flush(self, ep):
    self.rewards_per_ep = np.append(self.rewards_per_ep, np.array([[ep, self.cur_rewards]]), axis=0)
    self.cur_rewards = 0
    if len(self.cur_Qmx): Qmx_mean = np.mean(self.cur_Qmx)
    else: Qmx_mean = 0
    self.Qmx_per_ep = np.append(self.Qmx_per_ep, np.array([[ep, Qmx_mean]]), axis=0)
    self.cur_Qmx = np.array([])
  
  def save(self, name):
    pickle.dump(self.rewards_per_ep,open('./graphs/{}_rewards.pkl'.format(name),'wb'))
    pickle.dump(self.Qmx_per_ep,open('./graphs/{}_Qmx.pkl'.format(name),'wb'))

  def load(self,name):
    self.rewards_per_ep = pickle.load(open('./graphs/{}_rewards.pkl'.format(name),'rb'))
    self.Qmx_per_ep = pickle.load(open('./graphs/{}_Qmx.pkl'.format(name),'rb'))
  
  def plot(self, figsize=(12,10), rewards_avg_window_size=100):
    plt.clf()
    fig, axs = plt.subplots(1,2, figsize=(figsize))

    axs[0].plot(self.rewards_per_ep[:,0],self.rewards_per_ep[:,1], label="Episode Reward", linestyle="dotted", color='c')
    axs[0].plot(np.convolve(self.rewards_per_ep[:,0], np.ones(rewards_avg_window_size)/rewards_avg_window_size, mode="valid"),
                np.convolve(self.rewards_per_ep[:,1], np.ones(rewards_avg_window_size)/rewards_avg_window_size, mode="valid"),
                label="Moving Average", color='r')
    axs[0].set_title("Episode Reward")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].legend()

    axs[1].plot(self.Qmx_per_ep[:,0],self.Qmx_per_ep[:,1], color='c')
    axs[1].set_title("Average Max Q")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Q Value")

    plt.show()