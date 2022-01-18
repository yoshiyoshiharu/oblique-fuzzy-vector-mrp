import itertools
import numpy as np

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

  B_inv = np.linalg.inv(B)

  M = A @ B_inv

  M_inv = np.linalg.inv(M)

  delta_vertexes = []
  for v in itertools.product(delta_intervals[0], delta_intervals[1], delta_intervals[2], delta_intervals[3], delta_intervals[4], delta_intervals[5], delta_intervals[6], delta_intervals[7]):
    delta_vertexes.append(v)

  D_scenarios = []

  for delta_vertex in delta_vertexes:
    D_scenarios.append(np.round(M_inv @ delta_vertex).tolist())

  return D_scenarios
