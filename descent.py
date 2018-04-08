import numpy as np
import math

class Descent():
	m = 0
	n = 0
	EPS = 0.001
	step = 1
	mat = np.array((0,0))
	vec = np.array((0))

	def calc_p1(x, beta):
		#print(np.dot(x, beta))
		temp = np.dot(x, beta)
		if (temp >= 675):
			return 1.0
		return 1 / (1+math.exp(-temp))

	def calc_deri_1(beta):
		vec = np.zeros((Descent.n))
		for i in range(Descent.m):
			#print(Descent.calc_p1(Descent.mat[i], beta))
			vec += Descent.mat[i] * (Descent.calc_p1(Descent.mat[i], beta) - Descent.vec[i])
		return vec

	def close(x, y):
		for i in range(Descent.n):
			if (abs(x[i]-y[i]) > Descent.EPS):
				return False
		return True

	def recursion(beta):
		#print(Descent.calc_deri_2(beta))
		ret = beta - Descent.step * Descent.calc_deri_1(beta)
		print(ret)
		if (Descent.close(beta, ret)):
			return ret
		else:
			return Descent.recursion(ret)

	def solve(mat, vec):
		Descent.m = mat.shape[0]
		Descent.n = mat.shape[1]

		Descent.mat = mat
		Descent.vec = vec

		beta = np.ones((Descent.n))
		print(mat.shape)
		print(beta)
		return Descent.recursion(beta)		