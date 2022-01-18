# d = [[-1, 1, 0, 0 ...], ... [1, 1, 1 ...]] ^ -1 [[delta_l, delta_u], ...]を計算する

import numpy as np
T = 7
delta = np.array([
[6.5, 19.5] , [-12.0, -4.0] , [-136.5, -45.5] , [-40.5, -13.5] , [-64.5, -21.5] , [-49.5, -16.5] , [1392.5, 4177.5]
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
