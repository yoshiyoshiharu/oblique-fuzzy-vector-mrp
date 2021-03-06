import numpy as np
from scipy.optimize import linprog

P = 2
T = 4
R = 3

c_P = [1000, 500, 600, 100, 200, 100, 80, 100, 120, 100, 130, 120] # production cost of product p
b_P = [10000, 5000, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0] # sales price of product p 
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

def main(U):
  """-------------------??????-----------------------"""
  V_SIZE = [0] * P
  for p in range(P):
    V_SIZE[p] = sum(len(v) for v in U[p])

  print(f"U: {U}")
  print(f"V_SIZE : {V_SIZE}")

  """---------------------??????-----------------------"""
  # U[p][s][t]
  def index(current_p, current_s, current_t, U):
    V_SIZE = [0] * P
    for p in range(P):
      V_SIZE[p] = sum(len(v) for v in U[p])

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

  """---------------------????????????-----------------------"""
  # ????????????
  B = np.zeros(sum(V_SIZE))
  I = np.zeros(sum(V_SIZE))
  x = np.zeros(P * T)
  z = np.zeros(sum(V_SIZE))
  pi_s = np.zeros(P)
  pi = np.zeros(sum(V_SIZE))
  pi_t = np.ones(P)

  c = np.hstack([B, I, x, z, pi_s, pi, pi_t])

  # print(f"c: {c}")

  """---------------------?????????-----------------------"""
  # 1??????????????????
  A_eq = []
  b_eq = []

  # print("---------------------------------------1st constraint-------------------------------------------")
  for p in range(P):
    for s in range(len(U[p])):
      for t in range(T):
        pst = index(p, s, t, U)
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
        b_eq.append(U[p][s][t])

        # if p == 1 and t < 5:
        #   print(f"----------(p, s, w) = ({p}, {s}, {pst})----------" )
        #   debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
        #   print(f"b: {U[p][s][t]}")

  # 2??????????????????
  A_ub = []
  b_ub = []
  # print("------------------------------------2nd constraint (pi_s to pi_1)-------------------------------")
  # pi_s to pi_1
  for p in range(P):
    for s in range(len(U[p])):
      # initialize
      B = np.zeros(sum(V_SIZE))
      I = np.zeros(sum(V_SIZE))
      x = np.zeros(P * T)
      z = np.zeros(sum(V_SIZE))
      pi_s = np.zeros(P)
      pi = np.zeros(sum(V_SIZE))
      pi_t = np.zeros(P)

      pst = index(p, s, 0, U)

      I[pst] = c_I[p]
      B[pst] = c_B[p]

      pi_s[p] = 1
      pi[pst] = -1

      A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
      b_ub.append(0)

      # print(f"----------(p, s, t) = ({p}, {s}, {pst})----------" )
      # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

  # pi_1 to pi_T-1
  # print("------------------------------2nd constraint (pi_1 to pi_T-1)------------------------------")
  for p in range(P):
    for s in range(len(U[p])):
      for t in range(T - 2):
        # initialize
        B = np.zeros(sum(V_SIZE))
        I = np.zeros(sum(V_SIZE))
        x = np.zeros(P * T)
        z = np.zeros(sum(V_SIZE))
        pi_s = np.zeros(P)
        pi = np.zeros(sum(V_SIZE))
        pi_t = np.zeros(P)

        pstu = index(p, s, t, U)
        pstw = index(p, s, t + 1, U)

        I[pstw] = c_I[p]
        B[pstw] = c_B[p]

        pi[pstu] = 1
        pi[pstw] = -1

        A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
        b_ub.append(0)

        # print(f"----------(p, t, u, w) = ({p}, {t}, {pstu}, {pstw})----------" )
        # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

  # V_T-1 to V_T 3??????????????????
  # print("------------------------------3rd constraint (pi_T-1 to pi_T)------------------------------")
  for p in range(P):
    for s in range(len(U[p])):
      # initialize
      B = np.zeros(sum(V_SIZE))
      I = np.zeros(sum(V_SIZE))
      x = np.zeros(P * T)
      z = np.zeros(sum(V_SIZE))
      pi_s = np.zeros(P)
      pi = np.zeros(sum(V_SIZE))
      pi_t = np.zeros(P)

      pstu = index(p, s, T - 2, U)
      pstw = index(p, s, T - 1, U)

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

  # pi_u - pi_t 4??????????????????
  # print("----------------------------------------4th constraint----------------------------------------")
  for p in range(P):
    for s in range(len(U[p])):
      # initialize
      B = np.zeros(sum(V_SIZE))
      I = np.zeros(sum(V_SIZE))
      x = np.zeros(P * T)
      z = np.zeros(sum(V_SIZE))
      pi_s = np.zeros(P)
      pi = np.zeros(sum(V_SIZE))
      pi_t = np.zeros(P)

      pstu = index(p, s, T - 1, U)

      pi[pstu] = 1
      pi_t[p] = -1

      A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
      b_ub.append(0)

      # print(f"----------(p, t, w, u) = ({p}, {T}, {pstu}, t)----------" )
      # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

  # print("-------------------------------------5th constraint------------------------------------------")
  # pi_s = 0 5??????????????????
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

  # z_w <= v_w 6??????????????????
  # print("-------------------------------------6th constraint---------------------------------------")
  for p in range(P):
    for s in range(len(U[p])):
      # initialize
      B = np.zeros(sum(V_SIZE))
      I = np.zeros(sum(V_SIZE))
      x = np.zeros(P * T)
      z = np.zeros(sum(V_SIZE))
      pi_s = np.zeros(P)
      pi = np.zeros(sum(V_SIZE))
      pi_t = np.zeros(P)

      pst = index(p, s, T - 1, U)

      z[pst] = 1

      A_ub.append(np.hstack([B, I, x, z, pi_s, pi, pi_t]))
      b_ub.append(U[p][s][T-1])
      # print(f"----------(p, t, w, v) = ({p}, {T}, {pst}, {U[p][s][T-1]})----------" )
      # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

  # 7??????????????????
  # print("-------------------------------------7th constraint---------------------------------------")
  for p in range(P):
    for s in range(len(U[p])):
      # initialize
      B = np.zeros(sum(V_SIZE))
      I = np.zeros(sum(V_SIZE))
      x = np.zeros(P * T)
      z = np.zeros(sum(V_SIZE))
      pi_s = np.zeros(P)
      pi = np.zeros(sum(V_SIZE))
      pi_t = np.zeros(P)

      pst = index(p, s, T - 1, U)

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

  # x?????????
  # ???????????????
  # print("-------------------------------------resource constraint---------------------------------------")
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

  # ?????????????????????
  # print("-------------------------------------internal demand constraint---------------------------------------")
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

  # print("-------------------------------------Leadtime constraint---------------------------------------")
  # 0 ~ Ld???????????????????????????0??????????????????????????????
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

      p_i = p * T + i 
      x[p_i] = 1
      
      # print(f"----------(p, i) = ({p}, {i})----------" )
      # debug(np.hstack([B, I, x, z, pi_s, pi, pi_t]))

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

  """----------------------------LP??????------------------------------------"""
  res = linprog(c, A_eq = A_eq, b_eq = b_eq, A_ub = A_ub, b_ub = b_ub, bounds = bounds, method="revised simplex")
  x = list(map(round, res.x))

  print(f"fun: {res.fun}")
  debug(x)

  return res

"""------------------max S from fixed x---------------"""

def sub(x, delta_intervals, M):
  # print(f"M: {M}")
  print(f"delta_intervals: {delta_intervals}")

  for p in range(P):
    print(f"x[{p}]: {list(map(round, x[T * p:T * (p + 1)]))}")

  # ???????????????????????????????????????
  all_x_cost = 0
  for p in range(P):
    all_x_cost += c_P[p] * sum(x[T * p:T * (p + 1)])
  
  print(f"all_x_cost: {all_x_cost}")

  # ?????????????????????????????????????????????????????????
  c = []
  for p in range(P):
    for t in range(T):
      c.append(-c_B[p])
  for p in range(P):
    for t in range(T):
      c.append(-c_I[p])
  for p in range(P):
    for t in range(T):
      c.append(b_P[p])
  for p in range(P):
    for t in range(T):
      c.append(0) # D??????
  
  # print(f"c: {c}")

  """-------------------1st constraint-------------------"""
  # print("-------------------1st constraint-------------------")
  A_eq = []
  b_eq = []

  for p in range(P):
    for t in range(T):
      # initialize
      B = np.zeros(P * T)
      I = np.zeros(P * T)
      s = np.zeros(P * T)
      D = np.zeros(P * T)

      p_t = p * T + t

      B[p_t] = 1
      I[p_t] = -1 
      D[p_t] = -1

      X_int = 0 # ??????????????????
      for i in range(t + 1):
        p_i = p * T + i
        X_int += x[p_i]
        for j in range(P):
          j_i_Ldj = (i + Ld[j]) + j * T
          if i + Ld[j] < T:
            X_int += -b[p][j] * x[j_i_Ldj]
      
      A_eq.append(np.hstack([B, I, s, D]))
      b_eq.append(-X_int)

      # if p == 2 and t < 5:
      #   print(f"(p,t) = ({p},{t})")
      #   print(B)
      #   print(-X_int)

  """-------------------2nd constraint-------------------"""
  # print("-------------------2nd constraint-------------------")
  for p in range(P):
    for t in range(T):
      # initialize
      B = np.zeros(P * T)
      I = np.zeros(P * T)
      s = np.zeros(P * T)
      D = np.zeros(P * T)

      p_t = p * T + t

      B[p_t] = 1
      D[p_t] = -1
      for i in range(t + 1):
        p_i = p * T + i
        s[p_i] = 1

      A_eq.append(np.hstack([B, I, s, D]))
      b_eq.append(0)

  """-------------------demand constraint-------------------"""
  # print("-------------------demand constraint-------------------")
  # M(D_1, ..., D_T) \in [delta, delta]?????????
  # MD <= ??????
  A_ub = []
  b_ub = []

  for p in range(P):
    for t in range(T):
      # initialize
      B = np.zeros(P * T)
      I = np.zeros(P * T)
      s = np.zeros(P * T)
      D = np.zeros(P * T)
      
      D[p * T:p * T + T] = M[t]

      A_ub.append(np.hstack([B, I, s, D]))
      b_ub.append(delta_intervals[p][t][1]) # ??????

      # if p == 0:
      #   print(D)
      #   print(delta_intervals[p][t][1])

  # ?????? <= MD
  for p in range(P):
    for t in range(T):
      # initialize
      B = np.zeros(P * T)
      I = np.zeros(P * T)
      s = np.zeros(P * T)
      D = np.zeros(P * T)
      
      D[p * T:p * T + T] = -M[t]

      A_ub.append(np.hstack([B, I, s, D]))
      b_ub.append(-delta_intervals[p][t][0]) # ??????

      # if p == 0:
      #   print(D)
      #   print(-delta_intervals[p][t][0])


  B_bounds=[(0, None)] * (P * T)
  I_bounds=[(0, None)] * (P * T)
  s_bounds=[(0, None)] * (P * T)
  D_bounds=[(0, None)] * (P * T)

  bounds = B_bounds + I_bounds + s_bounds + D_bounds

  """----------------------------LP??????------------------------------------"""
  res = linprog(c, A_eq = A_eq, b_eq = b_eq, A_ub=A_ub, b_ub=b_ub, bounds = bounds, method="revised simplex")
  x = list(map(round, res.x))

  print(f"fun: {-res.fun + all_x_cost}")
  print(f"B: {x[0:P*T]}")
  print(f"I: {x[P*T:P*T + P*T]}")
  print(f"s: {x[P*T*2:P*T*2 + P*T]}")
  print(f"D: {x[P*T*3:P*T*3 + P*T]}")
  D = x[P*T*3:P*T*4]
  delta = []
  for p in range(P):
    D_p = D[p * T:p * T + T]
    delta.append(M @ D_p)
  
  print(f"delta: {delta}")
