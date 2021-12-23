import itertools
import numpy as np
from lib import worst_case_mrp

def worst_case_scenarios(T, delta_intervals):
  # -1, 1...のやつ
  A = np.zeros((T, T))
  for i in range(T - 1):
    A[i][i] = -1
    A[i][i + 1] = 1
  A[-1] = np.ones(T)

  # 1, 0, 0...のやつ
  B = np.zeros((T, T))
  for i in range(T):
    for j in range(i + 1):
      B[i][j] = 1

  M = A @ np.linalg.inv(B)
  M_inv = np.linalg.inv(M)

  delta_scenarios = []
  for v in itertools.product(delta_intervals[0], delta_intervals[1], delta_intervals[2], delta_intervals[3]):
    delta_scenarios.append(v)

  D_scenarios = []

  for delta_vertex in delta_scenarios:
    D_scenarios.append(M_inv @ delta_vertex)

  return D_scenarios
