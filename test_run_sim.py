from run_sim import *

C = Config()

C.G = .9
C.linear_net = False
C.learning_rate = 0.001
C.data_geometry = 'non_linear_fn'
C.L=10
C.print_progress = True
C.length_corridors = [10]*1
C.angle_range = np.pi/10
C.hidden_size = 250 # (C.length_corridors[0]+2*C.max_move+1 + 1)*len(C.length_corridors)
C.num_epochs = 10000
C.algo_name = 'SGD'
C.loss_fn = nn.MSELoss()

C.s_range = (-1.0, 1.0)
C.num_samples = 20
C.function_dim = 20
C.max_move = 0.2
C.discrete_samples = True

d = run_sim_wrapper(C)
