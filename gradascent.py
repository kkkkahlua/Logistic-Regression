import numpy as np
import math

class GradAscent():
	m = 0
	n = 0
	EPS = 0.001
	step = 0.001
	mat = np.array((0,0))
	vec = np.array((0))

	def sigmoid(x):
		for i in range(GradAscent.m):
			if (x[i]>=675):
				x[i] = 1
			if (x[i]<=-675):
				x[i] = 0
			else:
				x[i] = 1.0 / (1+math.exp(-x[i]))
		return x

	def close(x, y):
		for i in range(GradAscent.n):
			if (abs(x[i]-y[i]) > GradAscent.EPS):
				return False
		return True

	def recursion(beta):
		#print(GradAscent.calc_deri_2(beta))
		print(GradAscent.mat.shape)
		print(GradAscent.vec.shape)
		ret = beta + GradAscent.step * np.dot(GradAscent.mat.T, GradAscent.vec - GradAscent.sigmoid(np.dot(GradAscent.mat, beta)))
		print(ret)
		if (GradAscent.close(beta, ret)):
			return ret
		else:
			return GradAscent.recursion(ret)

	def solve(mat, vec):
		GradAscent.m = mat.shape[0]
		GradAscent.n = mat.shape[1]

		GradAscent.mat = mat
		GradAscent.vec = vec

		beta = np.ones((GradAscent.n))
		print(mat.shape)
		print(beta)
		return GradAscent.recursion(beta)		