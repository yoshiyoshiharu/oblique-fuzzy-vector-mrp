import itertools
import numpy as np
 
# ------ DATAS ------
P = 4
T = 4
R = 3

c_P = [1000, 600, 500, 100] # production cost of product p
b_P = [10000, 5500, 0, 0] # sales price of product p 
c_I = list(map(lambda x: x * 0.05, c_P)) # inventory cost of product p
c_B = list(map(lambda x: x * 0.15, b_P)) # backordering cost of product p

b = [[0, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 2, 0]] # b_(i,j)amount of product i to produce product j
Ld = [0, 0, 1, 0] # lead time of product p
a = [[1, 0, 1], [2, 0, 0], [0, 2, 0], [0, 0, 1]] # a_(p,r) amount of resource r to produce product p 
l = [[0, 0, 0], [0, 0 ,0], [1000, 2000, 2000], [1000, 2000, 2000]] # l_(t,r) lowwer resource r of period t
u = [[10000, 9100, 10300], [10000, 9100, 9300], [10000, 11200, 9300], [10000, 11200, 9300]] # u_(t,r)upper resource r of period t
L = [[0, 0, 0], [0, 0 ,0], [1000, 2000, 2000], [2000, 4000, 4000]]
U = [[10000, 9100, 10300], [20000, 18200, 19600], [30000, 29400, 28900], [40000, 40600, 38200]]

D = [[200, 300, 0, 0], [500, 600, 0, 0], [1000, 1200, 0, 0], [1500, 2000, 0, 0]]

c = c_I * T + c_B * T + c_P * T + list(map(lambda x: x * -1, b_P)) * T 


# ------ LP ------

# 制約式1のIの係数
I = [[0] * (T * P) for _ in range(T * P)]
for i in range(T * P):
  I[i][i] = -1

# 制約式1のBの係数
B = [[0] * (T * P) for _ in range(T * P)]
for i in range(T * P):
  B[i][i] = 1

# 1番目の制約式のxの係数
x = [[0] * (T * P) for _ in range(T * P)]

for t in range(T):
  # print(f"----------------------------t={t}-------------------------")
  for p in range(P):
    # print(f"------------------p={p}-------------------")
    t_p = t * P + p # t_p = 0, 1, 2, 3, 4 ,5 ... (t, p) = (2, 3)のとき 4を返す

    for i in range(t + 1):
      # print(f"--------------i={i}----------")
      i_p = i * P + p
      x[t_p][i_p] += 1
      for j in range(P):
        # print(f"--------j={j}------")
        i_Ldj_j = (i + Ld[j]) * P + j
        # print(i_Ldj_j)
        if i + Ld[j] < T:
          # print(f"(i+Ldj,j)=({i+Ld[j]}, {j}) b[p][j] = {b[p][j]}")
          x[t_p][i_Ldj_j] -= b[p][j]

s = [[0] * (T * P) for _ in range(T * P)]

A_1 = [[] * (T * P * 4) for _ in range(T * P)] # 1つ目の制約式の左辺
for i in range(T * P):
  A_1[i] = I[i] + B[i] + x[i] + s[i]

# sum s_(i,p) + B_(t, p)

I = [[0] * (T * P) for _ in range(T * P)]
B = [[0] * (T * P) for _ in range(T * P)]
x = [[0] * (T * P) for _ in range(T * P)]
s = [[0] * (T * P) for _ in range(T * P)]

for i in range(T * P):
  B[i][i] = 1

for t in range(T):
  for p in range(P):
    t_p = t * P + p

    for i in range(t + 1):
      i_p = i * P + p
      s[t_p][i_p] = 1

A_2 = [[] * (T * P * 4) for _ in range(T * P)] # 2つ目の制約式の左辺
for i in range(T * P):
  A_2[i] = I[i] + B[i] + x[i] + s[i]

A_eq = A_1 + A_2

b_eq = list(itertools.chain.from_iterable(D)) * 2


# 3つめの制約式 資源の制約 
ax = np.array([[0] * (T * P) for _ in range(T * R)])
# l < sum(ax) < u
for t in range(T):
  for r in range(R):
    t_r = t * R + r
    for j in range(P):
      t_j = t * P + j
      ax[t_r][t_j] = a[j][r]
  
AX = np.array([[0] * (T * P) for _ in range(T * R)])

for t in range(T):
  for r in range(R):
    t_r = t * R + r
    for j in range(P):
      for i in range(t + 1):
        i_j = i * P + j
        AX[t_r][i_j] = a[j][r]

print(AX)

zeros = np.zeros((T * R, T * P))
ax_ub = np.concatenate([zeros, zeros, ax, zeros], axis = 1) # よこにつなげる
AX_ub = np.concatenate([zeros, zeros, AX, zeros], axis = 1)

np.set_printoptions(threshold=10000)
ax_AX_ub = np.concatenate([ax_ub, AX_ub], axis = 0) # 縦につなげる

A_ub = np.concatenate([ax_AX_ub, ax_AX_ub * (-1)], axis = 0)

b_ub = np.ravel(np.concatenate([u, U, np.array(l) * (-1), np.array(L) * (-1)], axis = 0))

bounds =[
    (0, None) 
] * (T * P * 4)

from scipy.optimize import linprog
res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub = A_ub, b_ub = b_ub, bounds=bounds)
print(res)
