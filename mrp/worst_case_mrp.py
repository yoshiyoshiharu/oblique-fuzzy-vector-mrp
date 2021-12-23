from lib import worst_case_scenarios
import itertools
import numpy as np

"""---------------------初期データ-----------------------"""
P = 2
T = 2
R = 3

c_P = [1000, 600, 500, 100] # production cost of product p
b_P = [10000, 5500, 0, 0] # sales price of product p 
c_I = list(map(lambda x: x * 0.05, c_P)) # inventory cost of product p
c_B = list(map(lambda x: x * 0.15, b_P)) # backordering cost of product p

b = [[0, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 2, 0]] # b_(i,j)amount of product i to produce product j
Ld = [0, 0, 1, 0] # lead time of product p

a = [[1, 0, 1], [2, 0, 0], [0, 2, 0], [0, 0, 1]] # a_(p,r) amount of resource r to produce product p 
r_l = [[0, 0, 0], [0, 0 ,0], [0, 0, 0], [0, 0, 0]] # l_(t,r) lowwer resource r of period t
r_u = [[2000, 2000, 2000], [2000, 2000, 2000], [2000, 2000, 2000], [2000, 2000, 2000]] # u_(t,r)upper resource r of period t
r_L = [[0, 0, 0], [0, 0 ,0], [0, 0, 0], [0, 0, 0]]
r_U = [[2000, 2000, 2000], [4000, 4000, 4000], [6000, 6000, 6000], [8000, 8000, 8000]]

D_intervals = [
  [
    [100, 200],
    [400, 600],
    [800, 1000],
    [1000, 1200]
  ],

  [
    [200, 400],
    [600, 800],
    [1000, 1200],
    [1200, 1400]
  ]
]

"""-------------------準備-----------------------"""
V = [[[] for _ in range(T)] for _ in range(P)]

for p in range(len(D_intervals)):
  for i in range(T):
    for S in worst_case_scenarios.worst_case_scenarios(T, D_intervals[p]):
      if S[i] not in V[p][i]:
        V[p][i].append(S[i])
    V[p][i].sort()

for p in range(P):
  for v in V[p]:
    if not v:
      v.append(0)

V_SIZE = [0] * P
for p in range(P):
  V_SIZE[p] = sum(len(v) for v in V[p])

print(f"V : {V}")
"""---------------------関数-----------------------"""
def index(current_p, current_t, current_w, V):
  index = 0
  for p in range(current_p):
    for t in range(T):
      length = sum(len(v) for v in V[p])
    else:
      index += length

  for t in range(current_t):
    index += len(V[current_p][t])

  return index + current_w

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

print(f"c: {c}")

"""---------------------制約式-----------------------"""
# 1つ目の制約式
A_eq = []
b_eq = []

print("---------------------------------------1st constraint-------------------------------------------")
for p in range(P):
  for t in range(T):
    for w in range(len(V[p][t])):
      # print(p, t, w)
      ptw = index(p, t, w, V)
      # initialize
      B = np.zeros(sum(V_SIZE))
      I = np.zeros(sum(V_SIZE))
      x = np.zeros(P * T)
      z = np.zeros(sum(V_SIZE))
      pi_s = np.zeros(P)
      pi = np.zeros(sum(V_SIZE))
      pi_t = np.zeros(P)

      B[ptw] = 1
      I[ptw] = -1
      
      for i in range(t + 1):
        p_i = p * T + i
        x[p_i] += 1
        for j in range(P):
          j_i_Ldj = (i + Ld[j]) + j * T
          if i + Ld[j] < T:
            x[j_i_Ldj] -= b[p][j]
      
      A_eq.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
      b_eq.append(V[p][t][w])

      print(f"----------(p, t, w) = ({p}, {t}, {ptw})----------" )
      debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
      print(f"b: {V[p][t][w]}")

# 2つ目の制約式
A_ub = []
b_ub = []
print("------------------------------------2nd constraint (pi_s to pi_1)-------------------------------")
# pi_s to pi_1
for p in range(P):
  for w in range(len(V[p][0])):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    ptw = index(p, 0, w, V)

    I[ptw] = c_I[p]
    B[ptw] = c_B[p]

    pi_s[p] = 1
    pi[ptw] = -1

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(0)

    print(f"----------(p, t, w, u) = ({p}, 0, {ptw}, 0)----------" )
    debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# pi_1 to pi_T-1
print("------------------------------2nd constraint (pi_1 to pi_T-1)------------------------------")
for p in range(P):
  for t in range(T - 1):
    for (u_index, u_value), (w_index, w_value) in itertools.product(enumerate(V[p][t]), enumerate(V[p][t + 1])):
      # print(u_index, u_value, w_index, w_value)
      if u_value <= w_value:
        # initialize
        B = np.zeros(sum(V_SIZE))
        I = np.zeros(sum(V_SIZE))
        x = np.zeros(P * T)
        z = np.zeros(sum(V_SIZE))
        pi_s = np.zeros(P)
        pi = np.zeros(sum(V_SIZE))
        pi_t = np.zeros(P)

        ptu = index(p, t, u_index, V)
        ptw = index(p, t + 1, w_index, V)

        I[ptw] = c_I[p]
        B[ptw] = c_B[p]

        pi[ptu] = 1
        pi[ptw] = -1

        A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
        b_ub.append(0)

        print(f"----------(p, t, w, u) = ({p}, {t}, {ptu}, {ptw})----------" )
        debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# V_T-1 to V_T 3つめの制約式
print("------------------------------3rd constraint (pi_T-1 to pi_T)------------------------------")
for p in range(P):
  for (u_index, u_value), (w_index, w_value) in itertools.product(enumerate(V[p][T - 2]), enumerate(V[p][T - 1])):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    ptu = index(p, T - 2, u_index, V)
    ptw = index(p, T - 1, w_index, V)

    I[ptw] = c_I[p]
    B[ptw] = c_B[p]

    z[ptw] = -b_P[p]

    pi[ptu] = 1
    pi[ptw] = -1

    for i in range(T):
      p_i = p * T + i
      x[p_i] = c_P[p]

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(0)

    print(f"----------(p, t, u, w) = ({p}, {T-1}, {ptu}, {ptw})----------" )
    debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# pi_u - pi_t 4本目の制約式
print("----------------------------------------4th constraint----------------------------------------")
for p in range(P):
  for u in range(len(V[p][T - 1])):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    ptu = index(p, T - 1, u, V)

    pi[ptu] = 1
    pi_t[p] = -1

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(0)

    print(f"----------(p, t, w, u) = ({p}, {T}, {ptu}, t)----------" )
    debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

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
  debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# z_w <= v_w 6本目の制約式
print("-------------------------------------6th constraint---------------------------------------")
for p in range(P):
  for w in range(len(V[p][T-1])):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    ptw = index(p, T - 1, w, V)

    z[ptw] = 1

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(V[p][T - 1][w])
    print(f"----------(p, t, w, v) = ({p}, {T}, {ptw}, {V[p][T - 1][w]})----------" )
    debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# 7本目の制約式
print("-------------------------------------7th constraint---------------------------------------")
for p in range(P):
  for w in range(len(V[p][T-1])):
    # initialize
    B = np.zeros(sum(V_SIZE))
    I = np.zeros(sum(V_SIZE))
    x = np.zeros(P * T)
    z = np.zeros(sum(V_SIZE))
    pi_s = np.zeros(P)
    pi = np.zeros(sum(V_SIZE))
    pi_t = np.zeros(P)

    ptw = index(p, T - 1, w, V)

    z[ptw] = 1

    for i in range(T):
      p_i = p * T + i
      x[p_i] -= 1
      for j in range(P):
        j_i_Ldj = (i + Ld[j]) + j * T
        if i + Ld[j] < T:
          x[j_i_Ldj] += b[p][j]

    A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
    b_ub.append(0)
    print(f"----------(p, t, w) = ({p}, {T}, {ptw})------------" )
    debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

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
