P = 12
T = 23
R = 3

import numpy as np

"""---------------------初期データ-----------------------"""
P = 12
T = 23
R = 3

c_P = [1000, 500, 600, 100, 200, 100, 80, 100, 120, 100, 130, 120] # production cost of product p
b_P = [10000, 5500, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0] # sales price of product p 
c_I = list(map(lambda x: x * 0.05, c_P)) # inventory cost of product p
c_B = list(map(lambda x: x * 0.15, b_P)) # backordering cost of product p

b = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
] # b_(i,j)amount of product i to produce product j

Ld = [0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0] # lead time of product p

a = [
  [1, 0, 0], 
  [0, 2, 0], 
  [0, 0, 1], 
  [0, 2, 0],
  [0, 3, 0],
  [0, 0, 2],
  [0, 0, 2],
  [0, 0, 1],
  [0, 1, 0],
  [0, 1, 0],
  [0, 0, 1],
  [0, 0, 2]
] # a_(p,r) amount of resource r to produce product p 

r_l = [
  [0, 0, 0], 
  [0, 0 ,0], 
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000]
] # l_(t,r) lowwer resource r of period t
r_u = [
  [10000, 9100, 10300], 
  [10000, 9100, 10300], 
  [10000, 9100, 10300], 
  [10000, 9100, 10300], 
  [10000, 9100, 10300], 
  [10000, 9100, 10300], 
  [10000, 9100, 10300], 
  [10000, 9100, 10300], 
  [10000, 9100, 9300], 
  [10000, 9100, 9300], 
  [10000, 9100, 9300], 
  [10000, 9100, 9300], 
  [10000, 9100, 9300], 
  [10000, 9100, 9300], 
  [10000, 11200, 6800], 
  [10000, 11200, 6800], 
  [10000, 11200, 6800], 
  [10000, 11200, 10800], 
  [10000, 11200, 10800], 
  [10000, 11200, 10800], 
  [10000, 11200, 10800], 
  [10000, 11200, 10800], 
  [10000, 11200, 10800] 
] # u_(t,r)upper resource r of period t

d_nominals = [
  [0, 0, 457, 434, 511, 564, 540, 527, 622, 585, 550, 592, 527, 400, 420, 473, 473, 486, 478, 387, 360, 317, 284],
  [0, 0, 379, 386, 400, 319, 386, 300, 387, 154, 144, 166, 151, 191, 156, 115, 150, 182, 100, 140, 281, 263, 257]
]

"""---------シナリオ----------"""
# -1, 1...のやつ
A = np.zeros((T, T))
for i in range(T - 1):
  A[i][i] = -1
  A[i][i + 1] = 1
A[-1] = np.ones(T)

# 1, 0, 0...のやつ
B = np.zeros((T, T))
for i in range(T):
  for j in range(i + 1):
    B[i][j] = 1

B_inv = np.linalg.inv(B)

M = A @ B_inv
M_inv = np.linalg.inv(M)

delta_nominals = []
for d_nominal in d_nominals:
  array = []
  for i in range(len(d_nominal) - 1):
    array.append(d_nominal[i + 1] - d_nominal[i])
  array.append(sum(d_nominal))
  delta_nominals.append(array)

delta_intervals = []
for delta_nominal in delta_nominals:
  array = []
  for i in range(len(delta_nominal)):
    array.append([delta_nominal[i] - delta_nominal[i] * 0.5, delta_nominal[i] + delta_nominal[i] * 0.5])
  delta_intervals.append(array)


for p in range(P - len(delta_intervals)):
  array = [[0, 0]] * T
  delta_intervals.append(array)

# 初期値は全部下限
U = [[] for _ in range(P)]
for p in range(P):
  S_0 = []
  for i in range(T):
    S_0.append(delta_intervals[p][i][0])
  U[p].append(np.round(M_inv @ S_0))

oblique_robust(U)
