import numpy as np

# -1, 1... のやつ
def A_inv(T):
  A = list([0] * T for _ in range(T-1))
  A.append([1] * T)

  for i in range(T-1):
    A[i][i] = -1
    A[i][i+1] = 1

  # Aの逆行列
  A_inv = np.linalg.inv(A)
  return A_inv

def positive(A):
  P = [[0] * len(A) for _ in range(len(A))]
  for i in range(len(A)):
    for j in range(len(A)):
      P[i][j] = max(0, A[i][j])

  return P


def negative(A):
  N = [[0] * len(A) for _ in range(len(A))]
  for i in range(len(A)):
    for j in range(len(A)):
      N[i][j] = min(0, A[i][j])

  return N
