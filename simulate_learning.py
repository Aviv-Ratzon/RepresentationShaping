import numpy as np

L=60
A = 15

s = np.zeros((L, L))
a = np.zeros((2*A+1, L))

M = L**2 - int((L-A-1)*(L-A))
