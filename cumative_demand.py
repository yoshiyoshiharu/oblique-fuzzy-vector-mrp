import numpy as np
T = 5
delta = np.array([
  [1000, 1375], 
  [500, 1500], 
  [-1000, -500], 
  [-1750, -1000], 
  [8000, 10000]
])

delta_l = delta[:, 0]
delta_u = delta[:, 1]

cumulative = [[0] * T for _ in range(T)]
for i in range(T):
  for j in range(i + 1):
    cumulative[i][j] = 1

print(cumulative)
import p_n
P = np.array(p_n.positive(cumulative @ p_n.A_inv(T)))
N = np.array(p_n.negative(cumulative @ p_n.A_inv(T)))

# dの下限
D_l = P @ delta_l + N @ delta_u
print(D_l)

# dの上限
D_u = N @ delta_l + P @ delta_u
print(D_u)
