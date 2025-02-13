import matplotlib.pyplot as plt
import pickle

file_in = './graphs/rewards_hist.pkl'
file_out = './graphs/rewards_hist.png'

rewards_hist = pickle.load(open(file_in,'rb'))

plt.plot(rewards_hist[:,0],label='player1')
plt.plot(rewards_hist[:,1],label='player2')
plt.legend(loc='upper right')
plt.savefig(file_out)
plt.show()