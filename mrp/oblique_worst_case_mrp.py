import itertools
import numpy as np
from lib import oblique_worst_case_scenarios

"""---------------------初期データ-----------------------"""
P = 12
T = 8
R = 3

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

c_P = [1000, 500, 600, 100, 200, 100, 80, 100, 120, 100, 130, 120] # production cost of product p
b_P = [11000, 5500, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0] # sales price of product p 
c_B = list(map(lambda x: x * 0.15, b_P)) # backordering cost of product p

c_I = []

for i in range(P):
  cost = c_P[i] * 0.05
  for j in range(P):
    cost += 0.05 * b[j][i] * c_P[j]
  c_I.append(cost)

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
  [10000, 9100, 10300]
] # u_(t,r)upper resource r of period t

delta_intervals = [
  [
    [0.0, 0.0] , [228.5, 457.0] , [-34.5, -23.0] , [38.5, 77.0] , [26.5, 53.0] , [-36.0, -24.0] , [-19.5, -13.0] , [3033.0, 4549.5]
    # [-55.5, -18.5] , [-52.5, -17.5] , [21.0, 63.0] , [-97.5, -32.5] , [-190.5, -63.5] , [10.0, 30.0] , [26.5, 79.5] , [2084.5, 6253.5]
    # [6.5, 19.5] , [-12.0, -4.0] , [-136.5, -45.5] , [-40.5, -13.5] , [-64.5, -21.5] , [-49.5, -16.5] , [1392.5, 4177.5]
  ], 

  [
    [0.0, 0.0] , [189.5, 379.0] , [3.5, 7.0] , [7.0, 14.0] , [-121.5, -81.0] , [33.5, 67.0] , [-129.0, -86.0] , [2170.0, 3255.0]
    # [-349.5, -116.5] , [-15.0, -5.0] , [11.0, 33.0] , [-22.125, -7.5] , [40.0, 60.0] , [-35.0, -17.5] , [-41.0, -20.5] , [732.0, 2196.0]
    # [21.5, 48.0] , [-82.0, -41.0] , [20.0, 40.0] , [70.5, 141.0] , [-27.0, -18.0] , [-9.0, -3.0] , [686.5, 2059.5]
  ]
]

"""-------------------準備-----------------------"""
S = []
for p in range(len(delta_intervals)):
  S.append(oblique_worst_case_scenarios.worst_case_scenarios(T, delta_intervals[p]))

for p in range(P - len(delta_intervals)):
  S.append([np.zeros(T).tolist()])

V_SIZE = [0] * P
for p in range(P):
  V_SIZE[p] = sum(len(v) for v in S[p])

print(f"S : {S}")
print(f"V_SIZE : {V_SIZE}")

"""---------------------関数-----------------------"""
# S[p][s][t]
def index(current_p, current_s, current_t, S):
  V_SIZE = [0] * P
  for p in range(P):
    V_SIZE[p] = sum(len(v) for v in S[p])

  index = 0
  for p in range(current_p):
    index += V_SIZE[p]

  index += current_s * T

  return index + current_t

def debug(A):
  B = A[0:sum(V_SIZE)]
  I = A[sum(V_SIZE):sum(V_SIZE)*2]
  x = A[sum(V_SIZE)*2:sum(V_SIZE)*2 + (P * T)]
  z = A[sum(V_SIZE)*2 + (P * T):sum(V_SIZE)*3 + (P * T)]
  pi_s = A[sum(V_SIZE)*3 + (P * T):sum(V_SIZE)*3 + (P * T) + P]
  pi = A[sum(V_SIZE)*3 + (P * T) + P:sum(V_SIZE)*4 + (P * T) + P]
  pi_t = A[sum(V_SIZE)*4 + (P * T) + P:sum(V_SIZE)*4 + (P * T) + 2 * P]

  if not np.all(B == 0): print(f"B: {B}")
  if not np.all(I == 0): print(f"I: {I}")
  if not np.all(x == 0): print(f"x: {x}")
  if not np.all(z == 0): print(f"z: {z}")
  if not np.all(pi_s == 0): print(f"pi_s: {pi_s}")
  if not np.all(pi == 0): print(f"pi: {pi}")
  if not np.all(pi_t == 0): print(f"pi_t: {pi_t}")

"""--------------------------------LP-------------------------------------"""

"""---------------------目的関数-----------------------"""
# 目的関数
B = np.zeros(sum(V_SIZE))
I = np.zeros(sum(V_SIZE))
x = np.zeros(P * T)
z = np.zeros(sum(V_SIZE))
pi_s = np.zeros(P)
pi = np.zeros(sum(V_SIZE))
pi_t = np.ones(P)

c = np.hstack([B, I, x, z, pi_s, pi, pi_t])

# print(f"c: {c}")

"""---------------------制約式-----------------------"""
# 1つ目の制約式
A_eq = []
b_eq = []

print("---------------------------------------1st constraint-------------------------------------------")
for p in range(P):
  for s in range(len(S[p])):
    for t in range(T):
      pst = index(p, s, t, S)
      # initialize
      B = np.zeros(sum(V_SIZE))
      I = np.zeros(sum(V_SIZE))
      x = np.zeros(P * T)
      z = np.zeros(sum(V_SIZE))
      pi_s = np.zeros(P)
      pi = np.zeros(sum(V_SIZE))
      pi_t = np.zeros(P)

      B[pst] = 1
      I[pst] = -1
      
      for i in range(t + 1):
        p_i = p * T + i
        x[p_i] += 1
        for j in range(P):
          j_i_Ldj = (i + Ld[j]) + j * T
          if i + Ld[j] < T:
            x[j_i_Ldj] -= b[p][j]
      
      A_eq.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
      b_eq.append(S[p][s][t])

      # print(f"----------(p, s, w) = ({p}, {s}, {pst})----------" )
      # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
      # print(f"b: {S[p][s][t]}")

# 2つ目の制約式
A_ub = []
b_ub = []
print("------------------------------------2nd constraint (pi_s to pi_1)-------------------------------")
# pi_s to pi_1
for p in range(P):
  for s in range(len(S[p])):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    pst = index(p, s, 0, S)

    I[pst] = c_I[p]
    B[pst] = c_B[p]

    pi_s[p] = 1
    pi[pst] = -1

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(0)

    # print(f"----------(p, s, t) = ({p}, {s}, {pst})----------" )
    # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# pi_1 to pi_T-1
print("------------------------------2nd constraint (pi_1 to pi_T-1)------------------------------")
for p in range(P):
  for s in range(len(S[p])):
    for t in range(T - 2):
      # initialize
      B = np.zeros(sum(V_SIZE))
      I = np.zeros(sum(V_SIZE))
      x = np.zeros(P * T)
      z = np.zeros(sum(V_SIZE))
      pi_s = np.zeros(P)
      pi = np.zeros(sum(V_SIZE))
      pi_t = np.zeros(P)

      pstu = index(p, s, t, S)
      pstw = index(p, s, t + 1, S)

      I[pstw] = c_I[p]
      B[pstw] = c_B[p]

      pi[pstu] = 1
      pi[pstw] = -1

      A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
      b_ub.append(0)

      # print(f"----------(p, t, u, w) = ({p}, {t}, {pstu}, {pstw})----------" )
      # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# V_T-1 to V_T 3つめの制約式
print("------------------------------3rd constraint (pi_T-1 to pi_T)------------------------------")
for p in range(P):
  for s in range(len(S[p])):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    pstu = index(p, s, T - 2, S)
    pstw = index(p, s, T - 1, S)

    I[pstw] = c_I[p]
    B[pstw] = c_B[p]

    z[pstw] = -b_P[p]

    pi[pstu] = 1
    pi[pstw] = -1

    for i in range(T):
      p_i = p * T + i
      x[p_i] = c_P[p]

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(0)

    # print(f"----------(p, t, u, w) = ({p}, {T-1}, {pstu}, {pstw})----------" )
    # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# pi_u - pi_t 4本目の制約式
print("----------------------------------------4th constraint----------------------------------------")
for p in range(P):
  for s in range(len(S[p])):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    pstu = index(p, s, T - 1, S)

    pi[pstu] = 1
    pi_t[p] = -1

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(0)

    # print(f"----------(p, t, w, u) = ({p}, {T}, {pstu}, t)----------" )
    # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

print("-------------------------------------5th constraint------------------------------------------")
# pi_s = 0 5本目の制約式
for p in range(P):
  # initialize
  B = np.zeros(sum(V_SIZE))
  I = np.zeros(sum(V_SIZE))
  x = np.zeros(P * T)
  z = np.zeros(sum(V_SIZE))
  pi_s = np.zeros(P)
  pi = np.zeros(sum(V_SIZE))
  pi_t = np.zeros(P)

  pi_s[p] = 1

  A_eq.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
  b_eq.append(0)
  # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# z_w <= v_w 6本目の制約式
print("-------------------------------------6th constraint---------------------------------------")
for p in range(P):
  for s in range(len(S[p])):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    pst = index(p, s, T - 1, S)

    z[pst] = 1

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(S[p][s][T-1])
    # print(f"----------(p, t, w, v) = ({p}, {T}, {pst}, {S[p][s][T-1]})----------" )
    # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# 7本目の制約式
print("-------------------------------------7th constraint---------------------------------------")
for p in range(P):
  for s in range(len(S[p])):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    pst = index(p, s, T - 1, S)

    z[pst] = 1

    for i in range(T):
      p_i = p * T + i
      x[p_i] -= 1
      for j in range(P):
        j_i_Ldj = (i + Ld[j]) + j * T
        if i + Ld[j] < T:
          x[j_i_Ldj] += b[p][j]

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(0)
    # print(f"----------(p, t, w) = ({p}, {T}, {pst})------------" )
    # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# xの制約
# 資源の制約
for t in range(T):
  for r in range(R):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    for j in range(P):
      t_j = j * T + t
      x[t_j] = a[j][r]

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(r_u[t][r])

    # print(f"----------(t, r) = ({t}, {r})------------" )
    # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    # print(f"u: {r_u[t][r]}")

    x = np.zeros(P * T)
    for j in range(P):
      t_j = j * T + t
      x[t_j] = -a[j][r]

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(r_l[t][r])

    # print(f"----------(t, r) = ({t}, {r})------------" )
    # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    # print(f"l: {r_l[t][r]}")

# 内部需要の制約
for p in range(P):
  for t in range(T):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    for i in range(t + 1):
      p_i = p * T + i
      x[p_i] -= 1
      for j in range(P):
        j_i_Ldj = (i + Ld[j]) + j * T
        if i + Ld[j] < T:
          x[j_i_Ldj] += b[p][j]

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(0)

    # print(f"----------(p, t) = ({p}, {t})----------" )
    # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# 0 ~ Ldまではその生産量は0という制約式を加える
Ld_constraint = []
for p in range(P):
  for i in range(Ld[p]):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    p_i = p * P + i
    x[p_i] = 1

    A_eq.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_eq.append(0)

B_bounds=[(0, None)] * sum(V_SIZE)
I_bounds=[(0, None)] * sum(V_SIZE)
x_bounds=[(0, None)] * (P * T)
z_bounds=[(0, None)] * sum(V_SIZE)
pi_s_bounds=[(None, None)] * P
pi_bounds=[(None, None)] * sum(V_SIZE)
pi_t_bounds=[(None, None)] * P

bounds = B_bounds + I_bounds + x_bounds + z_bounds + pi_s_bounds + pi_bounds + pi_t_bounds

"""----------------------------LP解く------------------------------------"""
from scipy.optimize import linprog
res = linprog(c, A_eq = A_eq, b_eq = b_eq, A_ub = A_ub, b_ub = b_ub, bounds = bounds)
x = list(map(int, res.x))

print(f"目的関数値: {res.fun}")
debug(x)
