from monitor import Monitor
from game import Game

plot_name = 'duel_vbot_10000_vai_double_r_20000'

monitor = Monitor(Game.STATE_SHAPE,Game.ACTION_SHAPE)
monitor.load(plot_name)
monitor.plot()