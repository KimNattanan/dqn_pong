import trainer

trainer.train_bot(save_name='duel_vbot',
                  plot_name='duel_vbot_10000',
                  number_of_ep=10000,
                  display=True)

trainer.train_ai(save_name='duel_vbot_10000_vai_l', save_name2='duel_vbot_10000_vai_r',
                 checkpoint_path='./checkpoints/duel_vbot_10000.pt',
                 checkpoint_path2='./checkpoints/duel_vbot_10000.pt',
                 plot_name='duel_vbot_10000_vai_l_20000', plot_name2='duel_vbot_10000_vai_r_20000',
                 plot_load_name='duel_vbot_10000', plot_load_name2='duel_vbot_10000',
                 number_of_ep=20000, begin_ep=10001, start_epsilon=0.4,
                 display=True)
trainer.train_ai_double(save_name='duel_vbot_10000_vai_double_l', save_name2='duel_vbot_10000_vai_double_r',
                        checkpoint_path='./checkpoints/duel_vbot_10000.pt',
                        checkpoint_path2='./checkpoints/duel_vbot_10000.pt',
                        plot_name='duel_vbot_10000_vai_double_l_20000', plot_name2='duel_vbot_10000_vai_double_r_20000',
                        plot_load_name='duel_vbot_10000', plot_load_name2='duel_vbot_10000',
                        number_of_ep=20000,begin_ep=10001, start_epsilon=0.4,
                        display=True)

trainer.train_ai(save_name='duel_vai_l', save_name2='duel_vai_r',
                 plot_name='duel_vai_l', plot_name2='duel_vai_r',
                 number_of_ep=10000,
                 display=True)
trainer.train_ai_double(save_name='duel_vai_double_l', save_name2='duel_vai_double_r',
                        plot_name='duel_vai_double_l', plot_name2='duel_vai_double_r',
                        number_of_ep=10000,
                        display=True)