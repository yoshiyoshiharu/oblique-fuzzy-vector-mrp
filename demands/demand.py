# d = [[-1, 1, 0, 0 ...], ... [1, 1, 1 ...]] ^ -1 [[delta_l, delta_u], ...]を計算する

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

from lib import p_n
P = np.array(p_n.positive(p_n.A_inv(T)))
N = np.array(p_n.negative(p_n.A_inv(T)))

# dの下限
d_l = P @ delta_l + N @ delta_u
print(d_l)

# dの上限
d_u = N @ delta_l + P @ delta_u
print(d_u)