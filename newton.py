import numpy as np
import math

class Newton():
	m = 0
	n = 0
	EPS = 0.0001
	mat = np.array((0,0))
	vec = np.array((0))

	def calc_p1(x, beta):
		#print(np.dot(x, beta))
		#print(x, beta, np.dot(x, beta))
		temp = math.exp(np.dot(x, beta))
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
			mat += np.dot(c, r) * p1 * (1-p1)
		return mat

	def close(x, y):
		for i in range(Newton.n):
			if (abs(x[i]-y[i]) > Newton.EPS):
				return False
		return True

	def recursion(beta):
		ret = beta - np.dot(Newton.calc_deri_2(beta).T, Newton.calc_deri_1(beta))
		print(ret)
		if (Newton.close(beta, ret)):
			return ret
		else:
			return Newton.recursion(ret)

	def solve(mat, vec):
		Newton.m = mat.shape[0]
		Newton.n = mat.shape[1]
		Newton.mat = mat
		Newton.vec = vec
		beta = np.zeros((Newton.n))
		print(mat.shape)
		print(beta)
		return Newton.recursion(beta)