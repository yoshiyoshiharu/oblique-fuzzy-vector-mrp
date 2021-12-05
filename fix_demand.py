import numpy as np
from lib import p_n

T = 5

P = p_n.positive(p_n.A_inv(T))
N = p_n.negative(p_n.A_inv(T))

c = [1] * T * 2

delta = np.array([
  [1000, 2000], 
  [500, 1500], 
  [-1000, -500], 
  [-2000, -1000],
  [8000, 10000]
])

delta_l = delta[:, 0]
delta_u = delta[:, 1]

P = np.array(p_n.positive(p_n.A_inv(T)))
N = np.array(p_n.negative(p_n.A_inv(T)))

A = np.concatenate([P * (-1), N], axis=1)

b = P @ delta_l + N @ delta_u

bounds = [(0, None)] * T * 2

from scipy.optimize import linprog

res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
print(res)
