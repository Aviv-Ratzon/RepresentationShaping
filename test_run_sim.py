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
C.length_corridors = [10]*1
# C.input_size = 10
C.max_move = 5
C.hidden_size = 50 # (C.length_corridors[0]+2*C.max_move+1 + 1)*len(C.length_corridors)
C.num_epochs = 1
C.scalar_actions = False
C.cyclic_corridors = False
C.one_hot_inputs = False
C.one_hot_outputs = False
C.input_size = 15
C.output_size = 10


d = run_sim_wrapper(C)
X = d['X']
y = d['y']

print(X.shape)
print(y.shape)