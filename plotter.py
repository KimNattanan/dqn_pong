from monitor import Monitor
from game import Game

plot_name = 'duel_vbot_5000_vai_double_l_10000'

monitor = Monitor(Game.STATE_SHAPE,Game.ACTION_SHAPE)
monitor.load(plot_name)
monitor.plot()