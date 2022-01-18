import numpy as np
T = 8
delta = np.array([
[0.0, 0.0],
[228.5, 685.5],
[-34.5, -11.5],
[38.5, 115.5],
[26.5, 79.5],
[-36.0, -12.0],
[-19.5, -6.5],
[1516.5, 4549.5]
])
# [0.0, 0.0]
# [189.5, 568.5]
# [3.5, 10.5]
# [7.0, 21.0]
# [-121.5, -40.5]
# [33.5, 100.5]
# [-129.0, -43.0]
# [1085.0, 3255.0]
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
