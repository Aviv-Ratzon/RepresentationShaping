from run_sim import *


C = Config()
C.length_corridors = [10]
C.num_epochs = 100
C.corridor_dim = 1
C.max_move = 5
C.print_progress = True
C.mask_states = [(5,)]

d = run_sim_wrapper(C)
