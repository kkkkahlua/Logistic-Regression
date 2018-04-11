import numpy as np
import math
import random
from knn import KNN

class Smote():
	k = 10

	def pupulate(ret, mat, num, cur, vec, N, c):
		for i in range(N):
			idx = vec[random.randint(0, Smote.k-1)]
			for j in range(c):
				fac = random.random()
				ret[num+i][j] = mat[cur][j] + fac * (mat[idx][j] - mat[cur][j])


	def overSample(mat, T, N, n):
		if N <= 0:
			return mat

		ret = np.zeros(((T*N), n))
		tot = 0

		for i in range(T):
			vec = KNN.kNearestNeighbours(mat, T, n, i)
			Smote.pupulate(ret, mat, tot, i, vec, N, n)	
			tot += N

		return ret

	def genNew(matp, matn, c):
		mat = np.vstack((matp, matn))

		nump = matp.shape[0]
		numn = matn.shape[0]

		vec = np.hstack((np.ones((nump), dtype=int), np.zeros((numn), dtype=int)))
		
		if (nump > numn):
			ret = Smote.overSample(matn, numn, int(nump/(3*numn))-1, c)
			num = ret.shape[0]
			mat = np.vstack((mat, ret))
			vec = np.hstack((vec, np.zeros((num), dtype=int)))
		elif (nump < numn):
			ret = Smote.overSample(matp, nump, int(numn/(3*nump))-1, c)
			num = ret.shape[0]
			mat = np.vstack((mat, ret))
			vec = np.hstack((vec, np.ones((num), dtype=int)))

		return (mat, vec)