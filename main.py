import trainer

n = 1000

trainer.train_ai(save_name='vai_l', save_name2='vai_r',
                 plot_name='vai_l_{}'.format(n), plot_name2='vai_r_{}'.format(n),
                 number_of_ep=n,
                 display=False)
trainer.train_bot(save_name='vbot',
                  plot_name='vbot_{}'.format(n),
                  number_of_ep=n,
                  display=False)

trainer.train_ai(save_name='vbot_{}_vai_l'.format(n), save_name2='vbot_{}_vai_r'.format(n),
                 checkpoint_path='./checkpoints/vbot_{}.pt'.format(n),
                 checkpoint_path2='./checkpoints/vbot_{}.pt'.format(n),
                 plot_name='vbot_{}_vai_l_{}'.format(n,2*n), plot_name2='vbot_{}_vai_r_{}'.format(n,2*n),
                 plot_load_name='vbot_{}'.format(n), plot_load_name2='vbot_{}'.format(n),
                 number_of_ep=2*n, begin_ep=n+1, start_epsilon=0.2,
                 display=False)
trainer.train_ai_double(save_name='vbot_{}_vai_double_l'.format(n), save_name2='vbot_{}_vai_double_r'.format(n),
                        checkpoint_path='./checkpoints/vbot_{}.pt'.format(n),
                        checkpoint_path2='./checkpoints/vbot_{}.pt'.format(n),
                        plot_name='vbot_{}_vai_double_l_{}'.format(n,2*n), plot_name2='vbot_{}_vai_double_r_{}'.format(n,2*n),
                        plot_load_name='vbot_{}'.format(n), plot_load_name2='vbot_{}'.format(n),
                        number_of_ep=2*n,begin_ep=n+1, start_epsilon=0.2,
                        display=False)

trainer.train_ai_double(save_name='vai_double_l', save_name2='vai_double_r',
                        plot_name='vai_double_l_{}'.format(n), plot_name2='vai_double_r_{}'.format(n),
                        number_of_ep=n,
                        display=False)