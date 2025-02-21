from monitor import Monitor
from game import Game

plot_name = ''

monitor = Monitor(Game.STATE_SHAPE,Game.ACTION_SHAPE)
monitor.load(plot_name)
monitor.plot()