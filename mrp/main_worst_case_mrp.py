from lib import worst_case_scenarios
from lib import worst_case_mrp

T = worst_case_mrp.T
P = worst_case_mrp.P

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
