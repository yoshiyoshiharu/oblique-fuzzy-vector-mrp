import itertools

P = 3
T = 2
R = 2

c_I = [85, 30, 31] # inventory cost of product p
c_B = [1500, 825, 600] # backordering cost of product p
c_P = [1000, 500, 600] # production cost of product p
b_P = [10000, 5500, 0] # sales price of product p 
b = [[0, 0, 0], [1, 0, 0], [2, 0, 0]] # amount of product i to produce product j
Ld = [0, 0, 0] # lead time of product p
a = [[1, 0], [0, 2], [1, 0]] # a_(p,r) amount of resource r to produce product p 
l = [[1000, 2000], [1000, 2000], [1000, 2000]] # l_(t,r) lowwer resource r of period t
u = [[10000, 9100], [10000, 9100], [10000, 9100]] # u_(t,r)upper resource r of period t
L = [[1000, 2000], [2000, 4000], [3000, 6000]]
U = [[10000, 9100], [20000, 18200], [30000, 27300]]
  
d = [[100, 200], [350, 300], [350, 450]] # d_(t,p) demand of product p of period t
D = [[100, 200], [450, 500], [800, 950]]

c = c_I * T + c_B * T + c_P * T + list(map(lambda x: x * -1, b_P)) * T 

# B_t,p - I_t,p + sum(x_i,p - sum(b_p,j * x_i+Ldj,j)

I = [[0] * (T * P) for _ in range(T * P)]
for i in range(T * P):
  I[i][i] = -1

B = [[0] * (T * P) for _ in range(T * P)]
for i in range(T * P):
  B[i][i] = 1

x = [[0] * (T * P) for _ in range(T * P)]

for t in range(T):
  for p in range(P):
    t_p = t * P + p # t_p = 0, 1, 2, 3, 4 ,5 ... (t, p) = (2, 3)のとき 5を返す

    for i in range(t + 1):
      i_p = i * P + p
      x[t_p][i_p] += 1
      for j in range(P):
        i_Ldj_j = (i + Ld[j]) * P + j
        x[t_p][i_Ldj_j] -= b[p][j]

s = [[0] * (T * P) for _ in range(T * P)]

A_1 = [[] * (T * P * 4) for _ in range(T * P)] # 1つ目の制約式の左辺
for i in range(T * P):
  A_1[i] = I[i] + B[i] + x[i] + s[i]

# print(A_1)

# sum s_(i,p) + B_(t, p)

I = [[0] * (T * P) for _ in range(T * P)]
B = [[0] * (T * P) for _ in range(T * P)]
x = [[0] * (T * P) for _ in range(T * P)]
s = [[0] * (T * P) for _ in range(T * P)]

for i in range(T * P):
  B[i][i] = 1

for t in range(T):
  for p in range(P):
    t_p = t * P + p

    for i in range(t + 1):
      i_p = i * P + p
      s[t_p][i_p] = 1

A_2 = [[] * (T * P * 4) for _ in range(T * P)] # 2つ目の制約式の左辺
for i in range(T * P):
  A_2[i] = I[i] + B[i] + x[i] + s[i]

# print(A_2)

A = A_1 + A_2

b = list(itertools.chain.from_iterable(D)) * 2

ax = [[0] * (T * P) for _ in range(T * R)]
# l < sum(ax) < u
for t in range(T):
  for r in range(R):
    t_r = t * R + r
    for j in range(P):
      t_j = t * P + j
      ax[t_r][t_j] = a[j][r]

AX = [[0] * (T * P) for _ in range(T * R)]

AX = [[0] * (T * P) for _ in range(T * R)]
for t in range(T):
  for r in range(R):
    t_r = t * R + r
    for i in range(t + 1):
      for j in range(P):
        i_r = i * R + r
        t_j = t * P + j
        AX[i_r][t_j] += a[j][r]

print(AX)

bounds =[
    (0, None)     # -3 ≤ y ≤ ∞
] * (T * P * 4)


from scipy.optimize import linprog
res = linprog(c, A_eq=A, b_eq=b, bounds=bounds)

print(res)
