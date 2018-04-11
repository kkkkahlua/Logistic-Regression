import numpy as np
from heapq import heappush, heappop
import math
import random
from knn import KNN

class BorderlineSmote():
	k = 10

	def pupulate(ret, mat, num, cur, vec, N, c, set1):
		for i in range(N):
			idx = vec[random.randint(0, BorderlineSmote.k-1)]
			for j in range(c):
				flag = idx in set1
				fac = random.random()
				if not flag:
					fac = fac / 2
				ret[num+i][j] = mat[cur][j] + fac * (mat[idx][j] - mat[cur][j])

	def sampleType(v, mat, vec):
		pos = 0
		for i in range(BorderlineSmote.k):
			if vec[v[i]] == 1:
				pos += 1
		if (pos == BorderlineSmote.k):			#	noise
			return 2
		if (pos > int(BorderlineSmote.k / 2)):	#	danger
			return 1
		return 0								#	safe

	def overSample(matall, vecall, mat, T, tar, n):
		if tar <= 0:
			return mat

		danger = 0
		choice = np.zeros(T, dtype = int)

		set1 = set()

		for i in range(T):
			if i % 20 == 0:
				print('%d of %d (in finding danger set...)' % (i, T))
			vec = KNN.kNearestNeighbours(matall, matall.shape[0], n, i)
			typ = BorderlineSmote.sampleType(vec, matall, vecall)
			if typ == 1:
				choice[danger] = i
				danger += 1
				set1.add(i)

		N = int(tar / danger)
		tar = N * danger
		ret = np.zeros((tar, n))
		tot = 0		

		for i in range(danger):
			vec = KNN.kNearestNeighbours(mat, T, n, choice[i])
			BorderlineSmote.pupulate(ret, mat, tot, i, vec, N, n, set1)
			tot += N

		return ret

	def genNew(matp, matn, c):
		mat = np.vstack((matp, matn))

		nump = matp.shape[0]
		numn = matn.shape[0]

		vec = np.hstack((np.ones((nump), dtype=int), np.zeros((numn), dtype=int)))
		
		if (nump > numn):
			ret = BorderlineSmote.overSample(mat, vec, matn, numn, (int(nump/(3*numn))-1)*numn, c)
			num = ret.shape[0]
			mat = np.vstack((mat, ret))
			vec = np.hstack((vec, np.zeros((num), dtype=int)))
		elif (nump < numn):
			ret = BorderlineSmote.overSample(mat, vec, matp, nump, (int(numn/(3*nump))-1)*nump, c)
			num = ret.shape[0]
			mat = np.vstack((mat, ret))
			vec = np.hstack((vec, np.ones((num), dtype=int)))

		return (mat, vec)