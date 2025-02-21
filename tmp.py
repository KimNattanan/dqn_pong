from collections import deque
import numpy as np

a = deque()
a.append([1,9])
a.append([2,8])
a.append([3,7])
print(a)

b = np.array(a)
print(b)