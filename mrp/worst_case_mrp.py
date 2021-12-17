from lib import worst_case_scenarios
import itertools
import numpy as np


P = 4
T = 4
R = 3

c_P = [1000, 600, 500, 100] # production cost of product p
b_P = [10000, 5500, 0, 0] # sales price of product p 
c_I = list(map(lambda x: x * 0.05, c_P)) # inventory cost of product p
c_B = list(map(lambda x: x * 0.15, b_P)) # backordering cost of product p

b = [[0, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 2, 0]] # b_(i,j)amount of product i to produce product j
Ld = [0, 0, 1, 0] # lead time of product p

D_intervals = [
  [
    [100, 600],
    [400, 800],
    [500, 1000],
    [800, 1600]
  ],

  [
    [200, 400],
    [300, 1000],
    [900, 1500],
    [1900, 2100]
  ]
]

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

# V[p][t]
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
print("----------------------------------------------")


# 目的関数 B_w, I_w, x_t,p, z_w, v_w, pi_s. pi_w, pi_t
c = np.hstack([
 np.zeros(sum(V_SIZE)),
 np.zeros(sum(V_SIZE)),
 np.zeros(P * T),
 np.zeros(sum(V_SIZE)),
 np.zeros(sum(V_SIZE)),
 np.zeros(1),
 np.zeros(sum(V_SIZE)),
 np.ones(1)
])

# 1つ目の制約式
A_eq = []
for p in range(P):
  for t in range(T):
    for w in range(len(V[p][t])):
      # print(p, t, w)
      ptw = index(p, t, w, V)
      # initialize
      B = np.zeros(sum(V_SIZE))
      I = np.zeros(sum(V_SIZE))
      x = np.zeros(P * T)
      v = np.zeros(sum(V_SIZE))
      pi_s = np.zeros(1)
      pi_w = np.zeros(sum(V_SIZE))
      pi_t = np.zeros(1)

      B[ptw] = 1
      I[ptw] = -1
      
      for i in range(t + 1):
        p_i = p * T + i
        x[p_i] += 1
        for j in range(P):
          j_i_Ldj = (i + Ld[j]) + j * P
          if i + Ld[j] < T:
            x[j_i_Ldj] -= b[p][j]

      v[ptw] = -1
      
      A_eq.append(np.hstack([B, I, x, v, pi_s, pi_w, pi_t]))

b_eq = np.zeros(sum(V_SIZE))

# 2つ目の制約式
A_ub = []
for p in range(P):
  for t in range(T):
    for w in range(len(V[p][t])):
      ptw = index(p, t, w, V)
      # initialize
      B = np.zeros(sum(V_SIZE))
      I = np.zeros(sum(V_SIZE))
      x = np.zeros(P * T)
      v = np.zeros(sum(V_SIZE))
      pi_s = np.zeros(1)
      pi_w = np.zeros(sum(V_SIZE))
      pi_t = np.zeros(1)

      I[ptw] = c_I[p]
      B[ptw] = c_B[p]

A_s = [[] for _ in range(P)]
for p in range(P):
  for a in itertools.product([0], V[p][0]):
    A_s[p].append(a)

# print(A_s)

A = [[[] for _ in range(T - 1)] for _ in range(P)]
for p in range(P):
  for t in range(T - 1):
    for a in itertools.product(V[p][t], V[p][t + 1]):
      if a[0] <= a[1]:
        A[p][t].append(a)

for p in range(P):
    A[p].insert(0, A_s[p])


print(A)
