import numpy as np
T = 5

delta = np.array([
    [32.0, 48.0],
    [112.8, 169.2],
    [-21.6, -14.4],
    [-7.2, -4.8],
    [936.9, 1145.1]
])
delta_l = delta[:, 0]
delta_u = delta[:, 1]

cumulative = [[0] * T for _ in range(T)]
for i in range(T):
  for j in range(i + 1):
    cumulative[i][j] = 1

from lib import p_n
P = np.array(p_n.positive(cumulative @ p_n.A_inv(T)))
N = np.array(p_n.negative(cumulative @ p_n.A_inv(T)))

# Dの下限
D_l = P @ delta_l + N @ delta_u
print(D_l)

# Dの上限
D_u = N @ delta_l + P @ delta_u
print(D_u)

for i in range(T):
  print(f"& $[{D_l[i]}, {D_u[i]}]$")
