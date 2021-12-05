def worst_case_scenarios(T, D):
  # V_t,pを求める
  V = [[] for _ in range(T)]

  for i in range(T):
    V[i].append(D[i][0])

    for j in range(T):
      if D[i][0] < D[j][0] < D[i][1]:
        V[i].append(D[j][0])
      elif D[i][0] < D[j][1] < D[i][1]:
        V[i].append(D[j][1])
    
    V[i].append(D[i][1])

  import itertools

  D_S = [S for S in list(itertools.product(V[0], V[1], V[2], V[3])) if S[0] <= S[1] <= S[2] <= S[3]]

  return D_S
