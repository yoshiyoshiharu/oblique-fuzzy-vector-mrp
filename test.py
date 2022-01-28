d_nominals = [
  [0, 0, 457, 434, 511, 564, 540, 527, 622, 585, 550, 592, 527, 400, 420, 473, 473, 486, 478, 387, 360, 317, 284],
  [0, 0, 379, 386, 400, 319, 386, 300, 387, 154, 144, 166, 151, 191, 156, 115, 150, 182, 100, 140, 281, 263, 257]
]

print("--------t = T_1-------")
for d_nominal in d_nominals:
  for i in range(0, 5):
    delta = d_nominal[i + 1] - d_nominal[i]
    print(f"[{delta - abs(delta) * 0.2}, {delta + abs(delta) * 0.2}], ")
  else:
    D = sum(d_nominal[0:6])
    print(f"[{D - D * 0.1}, {D + D * 0.1}]")
  print("---------------")

print("--------t = T_2-------")
for d_nominal in d_nominals:
  for i in range(6, 11):
    delta = d_nominal[i + 1] - d_nominal[i]
    print(f"[{delta - abs(delta) * 0.2}, {delta + abs(delta) * 0.2}], ")
  else:
    D = sum(d_nominal[6:12])
    print(f"[{D - D * 0.1}, {D + D * 0.1}]")
  print("---------------")

print("--------t = T_3-------")
for d_nominal in d_nominals:
  for i in range(12, 17):
    delta = d_nominal[i + 1] - d_nominal[i]
    print(f"[{delta - abs(delta) * 0.2}, {delta + abs(delta) * 0.2}], ")
  else:
    D = sum(d_nominal[12:18])
    print(f"[{D - D * 0.1}, {D + D * 0.1}]")
  print("---------------")

print("--------t = T_4-------")
for d_nominal in d_nominals:
  for i in range(18, 22):
    delta = d_nominal[i + 1] - d_nominal[i]
    print(f"[{delta - abs(delta) * 0.2}, {delta + abs(delta) * 0.2}], ")
  else:
    D = sum(d_nominal[18:23])
    print(f"[{D - D * 0.1}, {D + D * 0.1}]")
  print("---------------")
