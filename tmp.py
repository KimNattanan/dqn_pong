import numpy as np

a = np.array([[1,2,3],[4,6,5],[9,8,7],[7,5,10]])
b = np.argmax(a,axis=1)

print(b)

c = np.array([[9,8,7],[6,5,4],[3,2,1],[10,11,12]])
d = c[np.arange(len(b)),b]

print(d)

# 7 5 3 12