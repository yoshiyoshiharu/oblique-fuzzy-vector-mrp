import numpy as np
T = 5
delta = np.array([
  [1000, 2000], 
  [500, 1500], 
  [-1500, -500], 
  [-2000, -1000], 
  [8000, 10000]
])

delta_l = delta[:, 0]
delta_u = delta[:, 1]

A = list([0] * T for _ in range(T-1))
A.append([1] * T)

for i in range(T-1):
  A[i][i] = -1
  A[i][i+1] = 1

# Aの逆行列
A_inv = np.linalg.inv(A)
P = [[0] * T for _ in range(T)]
N = [[0] * T for _ in range(T)]

for i in range(T):
  for j in range(T):
    P[i][j] = max(0, A_inv[i][j])
    N[i][j] = min(0, A_inv[i][j])

# dの下限
d_l = P @ delta_l + N @ delta_u
print(d_l)

# dの上限
d_u = N @ delta_l + P @ delta_u
print(d_u)
