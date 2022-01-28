import itertools
import numpy as np
from lib import oblique_worst_case_scenarios

"""---------------------初期データ-----------------------"""
P = 12
T = 5
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

Ld = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # lead time of product p

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
  # [0, 0, 0], 
  # [0, 0 ,0], 
  # [1000, 2000, 2000],
  # [1000, 2000, 2000],
  # [1000, 2000, 2000],
  # [1000, 2000, 2000],
  # [1000, 2000, 2000],
  # [1000, 2000, 2000],
  # [1000, 2000, 2000],
  # [1000, 2000, 2000],
  # [1000, 2000, 2000],
  # [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  [1000, 2000, 2000],
  # [1000, 2000, 2000],
  # [1000, 2000, 2000],
  # [1000, 2000, 2000],
  # [1000, 2000, 2000]
] # l_(t,r) lowwer resource r of period t
r_u = [
  # [10000, 9100, 10300], 
  # [10000, 9100, 10300], 
  # [10000, 9100, 10300], 
  # [10000, 9100, 10300], 
  # [10000, 9100, 10300], 
  # [10000, 9100, 10300], 
  # [10000, 9100, 10300], 
  # [10000, 9100, 10300], 
  # [10000, 9100, 9300], 
  # [10000, 9100, 9300], 
  # [10000, 9100, 9300], 
  # [10000, 9100, 9300], 
  # [10000, 9100, 9300], 
  # [10000, 9100, 9300], 
  # [10000, 11200, 6800], 
  # [10000, 11200, 6800], 
  # [10000, 11200, 6800], 
  # [10000, 11200, 10800], 
  [10000, 11200, 10800], 
  [10000, 11200, 10800], 
  [10000, 11200, 10800], 
  [10000, 11200, 10800], 
  [10000, 11200, 10800]
] # u_(t,r)upper resource r of period t

delta_intervals = [
  [
    # [0.0, 0.0],
    # [365.6, 457.0],
    # [-27.6, -23.0],
    # [61.6, 77.0],
    # [42.4, 53.0],
    # [1966.0, 2162.6]

    # [-15.6, -10.4],
    # [76.0, 114.0],
    # [-44.4, -29.6],
    # [-42.0, -28.0],
    # [33.6, 50.4],
    # [3074.4, 3757.6] 

    # [-152.4, -101.6],
    # [16.0, 24.0],
    # [42.4, 63.6],
    # [0.0, 0.0],
    # [10.4, 15.6],
    # [2501.1, 3056.9]

    [-109.2, -72.8],
    [-32.4, -21.6],
    [-51.6, -34.4],
    [-39.6, -26.4],
    [1643.4, 2008.6]

    ], 

  [
    # [0.0, 0.0],
    # [303.2, 379.0],
    # [5.6, 7.0],
    # [11.2, 14.0],
    # [-97.2, -81.0],
    # [1484.0, 1632.4]

    # [-103.2, -68.8],
    # [69.6, 104.4],
    # [-279.6, -186.4],
    # [-12.0, -8.0],
    # [17.6, 26.4],
    # [1383.3, 1690.7]

    # [32.0, 48.0],
    # [-42.0, -28.0],
    # [-49.2, -32.8],
    # [28.0, 42.0],
    # [25.6, 38.4],
    # [850.5, 1039.5]
    
    [32.0, 48.0],
    [112.8, 169.2],
    [-21.6, -14.4],
    [-7.2, -4.8],
    [936.9, 1145.1]
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

# print(f"S : {S}")
# print(f"V_SIZE : {V_SIZE}")

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

def debug_x(A):

  x = A[sum(V_SIZE)*2:sum(V_SIZE)*2 + (P * T)]

  for p in range(P):
    out = ""

    for value in x[p * T: p * T + T]:
      string = "$" + str(value) + "$"
      out += " & " + string
    print("$x_{t," + str(p + 1) + "}$")
    print(out + " \\\\")

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

      # if p == 0 and s == 0:
      #   print(f"----------(p, s, w) = ({p}, {s}, {pst})----------" )
      #   debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
      #   print(f"b: {S[p][s][t]}")

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

    # if p == 1 and s == 0:
    #   print(f"----------(p, s, t) = ({p}, {s}, {pst})----------" )
    #   debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

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

    # if p < 5:
    #   print(f"----------(p, t) = ({p + 1}, {t + 1})----------" )
    #   debug_x(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

# 0 ~ Ldまではその生産量は0という制約式を加える

# for p in range(P):
#   for i in range(Ld[p]):
#     # initialize
#     B = np.zeros(sum(V_SIZE))
#     I = np.zeros(sum(V_SIZE))
#     x = np.zeros(P * T)
#     z = np.zeros(sum(V_SIZE))
#     pi_s = np.zeros(P)
#     pi = np.zeros(sum(V_SIZE))
#     pi_t = np.zeros(P)

#     p_i = p * T + i
#     x[p_i] = 1

#     A_eq.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
#     b_eq.append(0)

B_bounds=[(0, None)] * sum(V_SIZE)
I_bounds=[(0, None)] * sum(V_SIZE)
x_bounds=[(0, None)] * (P * T)
z_bounds=[(0, None)] * sum(V_SIZE)
pi_s_bounds=[(0, None)] * P
pi_bounds=[(None, None)] * sum(V_SIZE)
pi_t_bounds=[(None, None)] * P

bounds = B_bounds + I_bounds + x_bounds + z_bounds + pi_s_bounds + pi_bounds + pi_t_bounds

"""----------------------------LP解く------------------------------------"""
from scipy.optimize import linprog
res = linprog(c, A_eq = A_eq, b_eq = b_eq, A_ub = A_ub, b_ub = b_ub, bounds = bounds)
x = []

for value in res.x:
  x.append(round(value, 1))

print(f"目的関数値: {res.fun}")
debug_x(x)
