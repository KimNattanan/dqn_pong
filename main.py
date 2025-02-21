import trainer

trainer.train_bot(save_name='duel_vbot', plot_name='duel_vbot', display=True)

trainer.train_ai(save_name='duel_vbot_20000_vai_l', save_name2='duel_vbot_20000_vai_r',
                 checkpoint_path='./checkpoints/duel_vbot_20000.pt',
                 checkpoint_path2='./checkpoints/duel_vbot_20000.pt',
                 plot_name='duel_vbot_20000_vai_l', plot_name2='duel_vbot_20000_vai_r',
                 plot_load_name='duel_vbot', plot_load_name2='duel_vbot',
                 number_of_ep=40000, begin_ep=20001, start_epsilon=0.4,
                 display=True)
trainer.train_ai_double(save_name='duel_vbot_20000_vai_double_l', save_name2='duel_vbot_20000_vai_double_r',
                        checkpoint_path='./checkpoints/duel_vbot_20000.pt',
                        checkpoint_path2='./checkpoints/duel_vbot_20000.pt',
                        plot_name='duel_vbot_20000_vai_double_l', plot_name2='duel_vbot_20000_vai_double_r',
                        plot_load_name='duel_vbot', plot_load_name2='duel_vbot',
                        number_of_ep=40000,begin_ep=20001, start_epsilon=0.4,
                        display=True)

trainer.train_ai(save_name='duel_vai_l', save_name2='duel_vai_r',
                 plot_name='duel_vai_l', plot_name2='duel_vai_r',
                 display=True)
trainer.train_ai_double(save_name='duel_vai_double_l', save_name2='duel_vai_double_r',
                        plot_name='duel_vai_double_l', plot_name2='duel_vai_double_r',
                        display=True)