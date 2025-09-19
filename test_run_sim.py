from run_sim import *
C = Config()


C.print_progress = True
C.G = 0.5
C.linear_net = True
C.split_actions = False
C.bias = False
# C.data_geometry = 'uneven_corridors'
# C.corridor_lengths = [30]
# C.corridor_widths = [2, 10]
C.learning_rate = 0.001
C.length_corridors = [10]*1
C.hidden_size = 100
C.L = 5
C.num_epochs = 10000
C.algo_name = 'Adam'
C.loss_fn = nn.CrossEntropyLoss()
C.max_move = 4
C.data_module = 'random_walk'
C.n_samples = 1000

d = run_sim_wrapper(C)
