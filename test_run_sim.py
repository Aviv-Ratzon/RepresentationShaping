from run_sim import *

C = Config()

C.G = 0.5
C.linear_net = False
C.learning_rate = 0.0001
C.data_geometry = 'non_linear_fn'
C.L=10
C.print_progress = True
C.length_corridors = [10]*1
C.angle_range = np.pi/10
C.hidden_size = 250 # (C.length_corridors[0]+2*C.max_move+1 + 1)*len(C.length_corridors)
C.num_epochs = 10000
C.algo_name = 'Adam'
C.loss_fn = nn.MSELoss()

C.s_range = (-1.0, 1.0)
C.poly_degree = 10
C.num_samples = 1000
C.function_dim = 1
C.max_move = 0.3
C.discrete_actions = True
C.n_actions = 10
C.discrete_samples = False
var_name = 'max_move'
var_values = np.logspace(-3, 0, 20)
# var_name = 'L'
# var_values = np.arange(1, 10)

d = run_sim_wrapper(C)
