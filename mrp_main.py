from lib import mrp
import numpy as np

res = mrp.mrp()

T = mrp.T
P = mrp.P
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
