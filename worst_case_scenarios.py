T = 4

D = [
  { 'lower': 50, 'upper': 250 },
  { 'lower': 100, 'upper': 300 },
  { 'lower': 150, 'upper': 350 },
  { 'lower': 200, 'upper': 400 },
]

# V_t,pを求める
V = [[] for _ in range(T)]

for i in range(T):
  V[i].append(D[i]['lower'])

  for j in range(T):
    if D[i]['lower'] < D[j]['lower'] < D[i]['upper']:
      V[i].append(D[j]['lower'])
    elif D[i]['lower'] < D[j]['upper'] < D[i]['upper']:
      V[i].append(D[j]['upper'])
  
  V[i].append(D[i]['upper'])

print(V)
