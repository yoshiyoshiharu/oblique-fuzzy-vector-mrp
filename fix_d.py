import numpy as np
import p_n

T = 4

P = p_n.positive(p_n.A_inv(T))
N = p_n.negative(p_n.A_inv(T))

c = [1] * 8

delta = np.array([
  [1000, 2000], 
  [500, 1500], 
  [-1500, -500], 
  [7500, 10000]
])

delta_l = delta[:, 0]
delta_u = delta[:, 1]

import p_n

P = np.array(p_n.positive(p_n.A_inv(T)))
N = np.array(p_n.negative(p_n.A_inv(T)))

A = np.concatenate([P * (-1), N], axis=1)

b = P @ delta_l + N @ delta_u

bounds = [
  (0, None),(0, None),(0, None),(0, None),(0, None),(0, None),(0, None),(0, None)
]

from scipy.optimize import linprog

res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
print(res)
