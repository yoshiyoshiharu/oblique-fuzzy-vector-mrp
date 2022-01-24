import oblique_robust
import numpy as np

P = oblique_robust.P
T = oblique_robust.T
R = oblique_robust.R

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

"""-------------solve robust--------------"""
res = oblique_robust.main(U)

V_SIZE = [0] * P
for p in range(P):
  V_SIZE[p] = sum(len(v) for v in U[p])

c1 = res.fun
x = res.x[sum(V_SIZE)*2:sum(V_SIZE)*2 + (P * T)]

# print(c1)
# print(list(map(round, x)))

# xをpずつに分割する

"""----------------max S from x^k --------------"""

oblique_robust.sub(x, delta_intervals, M)
