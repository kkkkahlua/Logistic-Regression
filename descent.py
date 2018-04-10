import numpy as np
import math

class Descent():
	m = 0
	n = 0
	EPS = 0.001
	DELTA = 0.001
	mat = np.array((0,0))
	vec = np.array((0))

	def p1(mat, beta):
		vec = np.dot(mat, beta)
	#	print('beta ', beta)
	#	print('vec ', vec)
		temp = math.e ** vec
	#	print('temp ', temp / (1 + temp))
		return temp / (1 + temp)

	def grad(beta):
		return np.dot(Descent.mat.T, (Descent.p1(Descent.mat, beta)-Descent.vec))

	def norm(beta):
		ret = 0
		for i in range(Descent.n):
			ret += pow(beta[i], 2)
		return math.sqrt(ret)

	def close(x, y):
		for i in range(Descent.n):
			if (abs(x[i]-y[i]) > Descent.EPS):
				return False
		return True

	def iteration():
		prev = np.ones((Descent.n))
		beta = np.zeros((Descent.n))
		cnt = 0
		while 1:
			cnt += 1
			grad = Descent.grad(beta)
			print('grad ', grad)
			print(Descent.norm(grad))
			if Descent.norm(grad) < Descent.EPS or Descent.close(beta, prev):
				print('grad ', grad)
				print(Descent.norm(grad))
				return beta
			prev = beta
			beta = beta - np.dot(Descent.DELTA, grad)
	#		print('\n')

	def solve(mat, vec):
		Descent.m = mat.shape[0]
		Descent.n = mat.shape[1]
		print(mat)
		Descent.mat = mat
		Descent.vec = vec

		return Descent.iteration()