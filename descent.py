import numpy as np
import math

class Descent():
	m = 0
	n = 0
	EPS = 1
	DELTA = 0.001
	mat = np.array((0,0))
	vec = np.array((0))

	def p1(mat, beta):
		vec = np.dot(mat, beta)
		print('beta ', beta)
		print('vec ', vec)
		temp = math.e ** vec
		print('temp ', temp / (1 + temp))
		return temp / (1 + temp)

	def grad(beta):
		return np.dot(Descent.mat.T, (Descent.p1(Descent.mat, beta)-Descent.vec))

	def norm(beta):
		ret = 0
		for i in range(Descent.n):
			ret += pow(beta[i], 2)
		return math.sqrt(ret)

	def iteration():
		beta = np.zeros((Descent.n))
		cnt = 0
		while 1:
			cnt += 1
			if cnt > 10:
				return
			grad = Descent.grad(beta)
			print('grad ', grad)
			print(Descent.norm(grad))
			if Descent.norm(grad) < Descent.EPS:
				return beta
			beta = beta - Descent.DELTA * grad
	#		print('\n')

	def solve(mat, vec):
		Descent.m = mat.shape[0]
		Descent.n = mat.shape[1]
		print(mat)
		Descent.mat = mat
		Descent.vec = vec

	#	return Descent.iteration()