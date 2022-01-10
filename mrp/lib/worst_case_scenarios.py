def worst_case_scenarios(T, D_interval):

  D = D_interval # [[D1の下限, D1の上限], [D2の下限, D2の上限], [D3の下限, D3の上限]....]
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

# T の 数やる
  D_S = [S for S in list(itertools.product(V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8], V[9], V[10], V[11], V[12], V[13], V[14], V[15], V[16], V[17], V[18], V[19], V[20], V[21], V[22])) if S[0] <= S[1] <= S[2] <= S[3] <=S[4] <= S[5] <= S[6] <= S[7] <=S[8] <= S[9] <= S[10] <= S[11] <=S[12] <= S[13] <= S[14] <= S[15] <=S[16] <= S[17] <= S[18] <= S[19] <=S[20] <= S[21] <= S[22]]

  return D_S
