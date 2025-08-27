from run_sim import *


C = Config()

C.seed = 0
# C.data_geometry = 'hyperbolic'
C.G = 0.5
C.linear_net = False
C.corridor_dim = 2
C.learning_rate = 0.1
C.L=1
C.print_progress = True
C.length_corridors = [5]*1
C.max_move = 1
C.hidden_size = 100
C.num_epochs = 10000
C.algo_name = 'SGD'
C.loss_fn = nn.CrossEntropyLoss()
# C.mask_states = [(1,1,)] #,(2,1,),(3,1,), (1,3,),(2,3,),(3,3,)]

d = run_sim_wrapper(C)
