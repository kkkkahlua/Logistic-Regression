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
		tmp = math.exp(np.dot(beta, x))
		return tmp / (1 + tmp)

	def calc_deri_1(beta):
		grad=np.zeros((DampedNewton.n))
		for i in range(DampedNewton.m):
		    grad+=np.dot((DampedNewton.calc_p1(beta,DampedNewton.mat[i])-DampedNewton.vec[i]),DampedNewton.mat[i])
		return grad

	def calc_deri_2(beta):
		Hessian=np.zeros((DampedNewton.n,DampedNewton.n))
		for i in range(DampedNewton.m):
			p1=DampedNewton.calc_p1(beta,DampedNewton.mat[i])
			Hessian+=(np.dot(DampedNewton.mat[i].T,DampedNewton.mat[i])*p1*(1-p1))
			#print(Hessian)
		return Hessian

	def close(x, y):
		return abs(x-y).all() < DampedNewton.EPS

	def ell(beta):
		ret=0
		for i in range(DampedNewton.m):
			ret+=(-DampedNewton.vec[i]*np.dot(beta,DampedNewton.mat[i])+math.log(1+math.exp(np.dot(beta,DampedNewton.mat[i]))))
		return ret		

	def get_alpha(beta,r,grad):
		delta = 0.5
		sigma = 0.25
		m=0
		while True:
			tmp=delta**m
			if DampedNewton.ell(beta+tmp*r)<=\
					DampedNewton.ell(beta)+np.dot(sigma*tmp,np.dot(grad,r)):
				return delta**m
			else:
				m += 1

	def recursion(beta, prev):
		deri_1 = DampedNewton.calc_deri_1(beta)
		deri_2 = DampedNewton.calc_deri_2(beta)
		print(deri_2)
		r = -np.dot(np.linalg.inv(deri_2), deri_1)

		ret = beta + DampedNewton.get_alpha(beta, r, deri_1) * r

		print(DampedNewton.norm(ret))
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
			#print(beta)
			print(norm)

			m = 0
			while 1:
				if m > DampedNewton.UPPER:
					return prev
				if not DampedNewton.ell(beta+pow(DampedNewton.DELTA, m) * r) <= DampedNewton.ell(beta) + DampedNewton.SIGMA * pow(DampedNewton.DELTA, m) * np.dot(deri_1, r):
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

#		return DampedNewton.iteration()
		
		beta = np.zeros((DampedNewton.n))
		print(mat.shape)
	#	print(beta)
		return DampedNewton.recursion(beta, beta)
		
		