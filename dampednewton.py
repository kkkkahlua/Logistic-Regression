import numpy as np
import math

class DampedNewton():
	m = 0
	n = 0
	cnt = 0

	EPS = 0.001
	DELTA = 0.5
	SIGMA = 0.25
	UPPER = 100

	
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
		vec = np.zeros((DampedNewton.n))
		for i in range(DampedNewton.m):
			#print(DampedNewton.calc_p1(DampedNewton.mat[i], beta))
			vec += np.dot(DampedNewton.mat[i], DampedNewton.calc_p1(DampedNewton.mat[i], beta) - DampedNewton.vec[i])
		return vec

	def calc_deri_2(beta):
		mat = np.zeros((DampedNewton.n, DampedNewton.n))
		c = np.zeros((DampedNewton.n, 1))
		r = np.zeros((1, DampedNewton.n))
		for i in range(DampedNewton.m):
			c[:,0] = DampedNewton.mat[i]
			r[0:] = DampedNewton.mat[i]
			p1 = DampedNewton.calc_p1(DampedNewton.mat[i], beta)
		#	print(p1)
			mat += np.dot(p1*(1-p1), np.dot(c, r))
		return mat

	def close(x, y):
		return abs(x-y).all() < DampedNewton.EPS

	def ell(beta):
		ret = 0
		for i in range(DampedNewton.m):
			temp = np.dot(beta, DampedNewton.mat[i])
			ret += -np.dot(temp, DampedNewton.vec[i])+ math.log(1+pow(math.e, temp), math.e)
		return ret

	def recursion(beta, prev):
		deri_1 = DampedNewton.calc_deri_1(beta)
		deri_2 = DampedNewton.calc_deri_2(beta)
		r = -np.dot(np.linalg.inv(deri_2), deri_1)

		m = 0
		while 1:
			if m > DampedNewton.UPPER:
				return prev
			if not DampedNewton.ell(beta+pow(DampedNewton.DELTA, m)) <= DampedNewton.ell(beta) + DampedNewton.SIGMA * pow(DampedNewton.DELTA, m) * np.dot(deri_1, r):
				break
			else:
				m += 1

		ret = beta + np.dot(pow(DampedNewton.DELTA, m), r)

		print(ret)
		if (DampedNewton.close(beta, ret)):
			return ret
		else:
			return DampedNewton.recursion(ret, beta)

	def norm(x):
		ret = 0
		for i in range(DampedNewton.n):
			ret += pow(x[i], 2)
		return math.sqrt(ret)

	def iteration():
		beta = np.zeros((DampedNewton.n))
		while 1:
			deri_1 = DampedNewton.calc_deri_1(beta)
			deri_2 = DampedNewton.calc_deri_2(beta)
			r = -np.dot(np.linalg.inv(deri_2), deri_1)
			norm = DampedNewton.norm(deri_1)		
			if DampedNewton.norm(deri_1) < DampedNewton.EPS:
				return beta
			print(beta)
			print(norm)

			m = 0
			while 1:
				if m > DampedNewton.UPPER:
					return prev
				if not DampedNewton.ell(beta+pow(DampedNewton.DELTA, m)) <= DampedNewton.ell(beta) + DampedNewton.SIGMA * pow(DampedNewton.DELTA, m) * np.dot(deri_1, r):
					break
				else:
					m += 1

			beta = beta + pow(DampedNewton.DELTA, m) * r


	def solve(mat, vec):
		DampedNewton.m = mat.shape[0]
		DampedNewton.n = mat.shape[1]
		DampedNewton.cnt = 0

		DampedNewton.mat = mat
		DampedNewton.vec = vec
		'''
		return DampedNewton.iteration()
		'''		
		beta = np.zeros((DampedNewton.n))
		print(mat.shape)
	#	print(beta)
		return DampedNewton.recursion(beta, beta)
		