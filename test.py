d_nominals = [
  [0, 0, 457, 434, 511, 564, 540, 527, 622, 585, 550, 592, 527, 400, 420, 473, 473, 486, 478, 387, 360, 317, 284],
  [0, 0, 379, 386, 400, 319, 386, 300, 387, 154, 144, 166, 151, 191, 156, 115, 150, 182, 100, 140, 281, 263, 257]
]

for d_nominal in d_nominals:
  print("---------------")
  for i in range(16, 22):
    delta = d_nominal[i + 1] - d_nominal[i]
    print(f"[{delta - abs(delta) * 0.5}, {delta + abs(delta) * 0.5}]")
  else:
    D = sum(d_nominal[16:23])
    print(f"[{D - D * 0.5}, {D + D * 0.5}]")
