from run_sim import *


C = Config()

C.G = 1
C.linear_net = True
C.learning_rate = 0.1
C.data_geometry = 'arm'
C.L=1
C.print_progress = True
C.length_corridors = [10]*1
C.max_move = np.pi/10
C.hidden_size = 50 # (C.length_corridors[0]+2*C.max_move+1 + 1)*len(C.length_corridors)
C.num_epochs = 10000
C.algo_name = 'SGD'
C.loss_fn = nn.MSELoss()

d = run_sim_wrapper(C)
