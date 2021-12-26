from lib import worst_case_scenarios
from lib import worst_case_mrp

T = worst_case_mrp.T
P = worst_case_mrp.P

D_intervals = [
  [
    [0, 1875],
    [2000, 4750],
    [5500, 8125],
    [8000, 10000]
  ], 
  [
    [250, 2000],
    [3500, 6000],
    [8250, 11000],
    [12000, 14000]
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

worst_case_mrp.worst_case_mrp(V)
