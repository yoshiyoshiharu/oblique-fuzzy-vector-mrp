from lib import oblique_worst_case_mrp
from lib import oblique_worst_case_scenarios
import numpy as np

T = oblique_worst_case_mrp.T
P = oblique_worst_case_mrp.P

# 修正後のdelta
delta_intervals = [
  [
    [1000, 2000],
    [500, 1500],
    [-1500, -1000],
    [8000, 10000]
  ], 

  [
    [2000, 3000],
    [1000, 1500],
    [-2000, -1000],
    [12000, 14000]
  ]
]

