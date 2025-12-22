from run_sim import *
C = Config()


C = Config()

C.G = 0.5
C.sig_h_2 = None
# C.gpu_id=1
# C.seed = 1
C.linear_net = True
# C.split_actions = True
# C.allow_backwards = True
C.learning_rate = 0.0001
C.L=5
C.print_progress = True
C.algo_name = 'Adam'
C.loss_fn = nn.CrossEntropyLoss()
C.length_corridors = [100]*1
# C.input_size = 10
C.max_move = 5
C.hidden_size = 50 # (C.length_corridors[0]+2*C.max_move+1 + 1)*len(C.length_corridors)
C.num_epochs = 1
C.scalar_actions = True
C.cyclic_corridors = False
C.one_hot_inputs = False
C.scalar_actions = True


d = run_sim_wrapper(C)
