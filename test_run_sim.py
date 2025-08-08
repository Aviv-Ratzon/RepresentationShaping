from run_sim import *


C = Config()

C.G = 1
C.linear_net = True
C.corridor_dim = 2
C.learning_rate = 0.005
C.L=10
C.print_progress = True
C.length_corridors = [5]*1
C.max_move = 2
C.hidden_size = 50
C.num_epochs *= 10
C.algo_name = 'SGD'
C.loss_fn = nn.CrossEntropyLoss()
C.mask_states = [(1,1,),(2,1,),(3,1,), (1,3,),(2,3,),(3,3,)]

d = run_sim_wrapper(C)
