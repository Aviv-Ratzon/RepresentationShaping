from run_sim import *
C = Config()


C = Config()

C = Config()

C.G = 0.5
C.sig_h_2 = None
C.linear_net = True
C.learning_rate = 0.0001
C.L=3
C.print_progress = True
C.algo_name = 'Adam'
C.loss_fn = nn.CrossEntropyLoss()
C.length_corridors = [20]*1
# C.input_size = 10
C.hidden_size = 100 # (C.length_corridors[0]+2*C.max_move+1 + 1)*len(C.length_corridors)
C.num_epochs = 10000
C.mix_inputs = 0.5
C.mix_outputs = 0.5
C.max_move = 10


d = run_sim_wrapper(C)
X = d['X']
y = d['y']

print(X.shape)
print(y.shape)