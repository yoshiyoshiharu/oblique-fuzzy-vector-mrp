from lib import mrp
from lib import worst_case_scenarios
import numpy as np

T = mrp.T
P = mrp.P

D1_interval = [
  [100, 300],
  [400, 600],
  [900, 1100],
  [1400, 1600]
]

D2_interval = [
  [200, 400],
  [500, 700],
  [1100, 1300],
  [1900, 2100]
]

D1_worst_case_scenarios = np.array(worst_case_scenarios.worst_case_scenarios(T, D1_interval))
D2_worst_case_scenarios = np.array(worst_case_scenarios.worst_case_scenarios(T, D2_interval))

min_res_fun = 0
for D1_scenario in D1_worst_case_scenarios:
  for D2_scenario in D2_worst_case_scenarios:
    D_S = np.vstack([D1_scenario, D2_scenario])

    zeros = np.zeros((P - len(D_S), P))
    D = np.vstack([D_S, zeros]).T
    res = mrp.mrp(D)

    if min_res_fun > res.fun:
      min_res = res
      min_res_fun = res.fun

x = np.array(min_res.x).reshape(T*4, P)

print(f"目的関数値: {min_res.fun}")

print("---------------I---------------")
print(x[0:T])

print("---------------B---------------")
print(x[T:2*T])

print("---------------x---------------")
print(x[2*T:3*T])

print("---------------s---------------")
print(x[3*T:4*T])
