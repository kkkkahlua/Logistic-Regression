import numpy as np
import math

class Newton():
	m = 0
	n = 0
	cnt = 0
	EPS = 0.001
	upper = 100
	mat = np.array((0,0))
	vec = np.array((0))

	def calc_p1(x, beta):
		#print('x', x, 'beta', beta)
		temp = np.dot(x, beta)
		if (temp >= 675):
			return 1.0
		temp = math.exp(temp)
		return temp / (1+temp)

	def calc_deri_1(beta):
		vec = np.zeros((Newton.n))
		for i in range(Newton.m):
			#print(Newton.calc_p1(Newton.mat[i], beta))
			vec += Newton.mat[i] * (Newton.calc_p1(Newton.mat[i], beta) - Newton.vec[i])
		return vec

	def calc_deri_2(beta):
		mat = np.zeros((Newton.n, Newton.n))
		c = np.zeros((Newton.n, 1))
		r = np.zeros((1, Newton.n))
		for i in range(Newton.m):
			c[:,0] = Newton.mat[i]
			r[0:] = Newton.mat[i]
			p1 = Newton.calc_p1(Newton.mat[i], beta)
		#	print(p1)
			mat += np.dot(c, r) * p1 * (1-p1)
		return mat

	def close(x, y):
		return abs(x-y).all() < Newton.EPS

	def recursion(beta):
		deri_1 = Newton.calc_deri_1(beta)
		deri_2 = Newton.calc_deri_2(beta)
		ret = beta - np.dot(np.linalg.inv(deri_2), deri_1)
		print(ret)
		if (Newton.close(beta, ret)):
			return ret
		else:
			return Newton.recursion(ret)

	def solve(mat, vec):
		Newton.m = mat.shape[0]
		Newton.n = mat.shape[1]
		Newton.cnt = 0

		Newton.mat = mat
		Newton.vec = vec
		beta = np.zeros((Newton.n))
		print(mat.shape)
		print(beta)
		return Newton.recursion(beta)