from run_sim import *


C = Config()
C.length_corridors = [3]
C.num_epochs = 100
C.corridor_dim = 2
C.max_move = 1
C.print_progress = True

d = run_sim_wrapper(C)
