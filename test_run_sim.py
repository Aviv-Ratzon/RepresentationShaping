from run_sim import *
C = Config()

C.G = 0.4
C.linear_net = False
C.learning_rate = 0.0001
C.data_geometry = 'non_linear_fn'
C.L=10
C.print_progress = True
C.length_corridors = [10]*1
C.angle_range = np.pi/10
C.hidden_size = 50 # (C.length_corridors[0]+2*C.max_move+1 + 1)*len(C.length_corridors)
C.num_epochs = 10000
C.algo_name = 'Adam'
C.loss_fn = nn.MSELoss()

C.s_range = (-2.0, 2.0)
C.num_samples = 10000
C.function_dim = 2
C.max_move = 0.3
C.discrete_actions = False
C.B = 0.1
C.n_actions = 5
C.discrete_samples = False
C.continuous_function = False
C.n_breakpoints = 5
var_name = 'max_move'
var_values = np.linspace(0., 2, 10)
# var_name = 'L'
# var_values = np.arange(1, 10)
d = run_sim_wrapper(C)
