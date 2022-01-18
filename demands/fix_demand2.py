import numpy as np
from lib import p_n

T = 8

P = p_n.positive(p_n.A_inv(T))
N = p_n.negative(p_n.A_inv(T))

# e_1^-, ..., e_T^-, e_1^+,...,e_T^+
c = np.hstack([np.zeros(T), np.zeros(T), np.ones(1)])

delta = np.array([
  [0.0, 0.0] , [228.5, 685.5] , [-34.5, -11.5] , [38.5, 115.5] , [26.5, 79.5] , [-36.0, -12.0] , [-19.5, -6.5]  , [1516.5, 4549.5]
])

delta_l = delta[:, 0]
delta_u = delta[:, 1]

P = np.array(p_n.positive(p_n.A_inv(T)))
N = np.array(p_n.negative(p_n.A_inv(T)))
minus_P = (-1) * P

b = P @ delta_l + N @ delta_u


A_ub = []
b_ub = []
for i in range(T):
  # initialize
  e_minus = np.zeros(T)
  e_plus = np.zeros(T)
  v = np.zeros(1)

  e_minus = minus_P[i]
  e_plus = N[i]
  A_ub.append(np.hstack([e_minus, e_plus, v]))
  b_ub.append(b[i])

for i in range(T):
  # initialize
  e_minus = np.zeros(T)
  e_plus = np.zeros(T)
  v = np.zeros(1)

  e_minus[i] = 1
  v[0] = -1
  A_ub.append(np.hstack([e_minus, e_plus, v]))
  b_ub.append(0)

for i in range(T):
  # initialize
  e_minus = np.zeros(T)
  e_plus = np.zeros(T)
  v = np.zeros(1)

  e_plus[i] = 1
  v[0] = -1
  A_ub.append(np.hstack([e_minus, e_plus, v]))
  b_ub.append(0)

bounds = []
for i in range(T):
  half_width = (delta_u[i] - delta_l[i])/2
  bounds.append((0, half_width))

for i in range(T, 2 * T):
  half_width = (delta_u[i - T] - delta_l[i - T])/2
  bounds.append((0, half_width))

bounds.append([0, None])
from scipy.optimize import linprog

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="revised simplex")
x = res.x

# for i in range(T):
#   print(f"e_{i + 1}^- = {x[i]}, e_{i + 1}^+ = {x[i + T]}")
# print(f"v = {x[2 * T]}")
  
for i in range(T):
  print(f" [{delta_l[i] + x[i]}, {delta_u[i] - x[i + T]}]")
