from lib import mrp
from lib import worst_case_scenarios
import numpy as np

if __name__ == '__main__':
  T = mrp.T
  P = mrp.P

  D_interval = []
  worst_case_scenarios = worst_case_scenarios.worst_case_scenarios(T, D_interval)


  D = [[200, 300, 0, 0], [500, 600, 0, 0], [1000, 1200, 0, 0], [1500, 2000, 0, 0]]
  res = mrp.mrp(D)

  x = np.array(res.x).reshape(T*4, P)

  print(f"目的関数値: {res.fun}")

  print("---------------I---------------")
  print(x[0:T])

  print("---------------B---------------")
  print(x[T:2*T])

  print("---------------x---------------")
  print(x[2*T:3*T])

  print("---------------s---------------")
  print(x[3*T:4*T])
