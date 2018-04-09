from train import Train
from test import Test
import numpy as np

def main():
	n = 11
	k = 5

	train = Train()
	test = Test()
	
	beta = np.zeros((k, n))

	for i in range(k):
		beta[i] = train.solve(i+1)
		print(beta[i])

	'''
	beta[0] = train.solve(4)

	print(beta[0])
	'''
	test.predict(beta, k)

main()