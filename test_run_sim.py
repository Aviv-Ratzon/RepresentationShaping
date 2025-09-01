from run_sim import *

C = Config()

C.G = 1
C.linear_net = False
C.learning_rate = 0.0001
C.data_geometry = 'MNIST'
C.L=5
C.print_progress = True
C.length_corridors = [10]*1
C.angle_range = np.pi/10
C.hidden_size = 1000 # (C.length_corridors[0]+2*C.max_move+1 + 1)*len(C.length_corridors)
C.num_epochs = 100000
C.algo_name = 'Adam'
C.loss_fn = nn.MSELoss()

C.s_range = (-1.0, 1.0)
C.poly_degree = 10
C.num_samples = 10000
C.B = 0.1
C.function_dim = 20
# C.max_move = 0.3
# C.discrete_samples = True
var_name = 'max_move'
var_values = [0, 1, 5] # np.linspace(1, 4, 10)
# var_name = 'L'
# var_values = np.arange(1, 3)

d = run_sim_wrapper(C)
