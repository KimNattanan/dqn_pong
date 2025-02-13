import numpy as np

c = np.array([]).reshape(0,2)
a = np.array([1,2])
b = np.array([3,4])
a+=b
print(a)
print(c)
c = np.append(c,[a])
c = np.append(c,[b])
c = np.append(c,[a])
c = np.append(c,[a])
print(c)