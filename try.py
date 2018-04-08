from heapq import heappush, heappop
import numpy as np

a = np.zeros((0,3))
b = np.zeros((4,3))
b[1] = [2,3,4]
a = np.row_stack((a,b))
c = a[1:3]
print(a)
print(c)
print(c.shape)