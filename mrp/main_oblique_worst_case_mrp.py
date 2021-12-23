from lib import worst_case_mrp
from lib import oblique_worst_case_scenarios
import numpy as np

T = worst_case_mrp.T
P = worst_case_mrp.P

# 修正後のdelta
delta_intervals = [
  [
    [1000, 1375],
    [500, 1500],
    [-1000, -500],
    [8000, 10000]
  ]
]

for p in range(len(delta_intervals)):
  D_scenarios = oblique_worst_case_scenarios.worst_case_scenarios(T, delta_intervals[p])
  print(D_scenarios)

# D_intervals = [
#   [
#     [100, 600],
#     [400, 800],
#     [500, 1000],
#     [800, 1600]
#   ],
#   [
#     [200, 400],
#     [300, 1000],
#     [900, 1500],
#     [1900, 2100]
#   ]
# ]

"""-------------------準備-----------------------"""
V = [[[] for _ in range(T)] for _ in range(P)]

for p in range(len(delta_intervals)):
  for i in range(T):
    for S in oblique_worst_case_scenarios.worst_case_scenarios(T, delta_intervals[p]):
      if S[i] not in V[p][i]:
        V[p][i].append(S[i])
    V[p][i].sort()

for p in range(P):
  for v in V[p]:
    if not v:
      v.append(0)

print(V)
worst_case_mrp.worst_case_mrp(V)
