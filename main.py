import trainer

trainer.train_bot(save_name='duel_vbot')
trainer.train_ai(save_name='duel_vai_l',save_name2='duel_vai_r')
trainer.train_ai_double(save_name='duel_vai_double_l',save_name2='duel_vai_double_r')

trainer.train_ai(save_name='duel_vbot_20000_vai_l', save_name2='duel_vbot_20000_vai_r',
                 checkpoint_path='./checkpoints/duel_vbot_20000.pt',
                 checkpoint_path2='./checkpoints/duel_vbot_20000.pt',
                 begin_ep=20001, start_epsilon=0.4)
trainer.train_ai_double(save_name='duel_vbot_20000_vai_double_l', save_name2='duel_vbot_20000_vai_double_r',
                        checkpoint_path='./checkpoints/duel_vbot_20000.pt',
                        checkpoint_path2='./checkpoints/duel_vbot_20000.pt',
                        begin_ep=20001, start_epsilon=0.4)