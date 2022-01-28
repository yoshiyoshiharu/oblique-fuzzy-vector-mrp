import numpy as np
from lib import p_n

T = 5

P = p_n.positive(p_n.A_inv(T))
N = p_n.negative(p_n.A_inv(T))

# e_1^-, ..., e_T^-, e_1^+,...,e_T^+
c = [1] * T * 2

delta = np.array([
# [-109.2, -72.8], 
# [-32.4, -21.6], 
# [-51.6, -34.4], 
# [-39.6, -26.4], 
# [1643.4, 2008.6]
# ---------------
[32.0, 48.0], 
[112.8, 169.2], 
[-21.6, -14.4], 
[-7.2, -4.8], 
[936.9, 1145.1]
])

delta_l = delta[:, 0]
delta_u = delta[:, 1]

P = np.array(p_n.positive(p_n.A_inv(T)))
N = np.array(p_n.negative(p_n.A_inv(T)))

A = np.concatenate([P * (-1), N], axis=1)

b = P @ delta_l + N @ delta_u

bounds = []
# 差異変数は幅の半分以下
for i in range(T):
  half_width = (delta_u[i] - delta_l[i])/2
  bounds.append((0, half_width))

for i in range(T, 2 * T):
  half_width = (delta_u[i - T] - delta_l[i - T])/2
  bounds.append((0, half_width))

from scipy.optimize import linprog

res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="revised simplex")
x = res.x

# for i in range(T):
#   print(f"e_{i + 1}^- = {x[i]}, e_{i + 1}^+ = {x[i + T]}")
  
for i in range(T):
  print(f" [{delta_l[i] + x[i]}, {delta_u[i] - x[i + T]}]")
