from train import Train
from test import Test
import numpy as np

n = 11
k = 5

train = Train()
test = Test()

beta = np.zeros((k, n))

for i in range(k):
	beta[i] = train.solve(i+1)

test.predict(beta, k)