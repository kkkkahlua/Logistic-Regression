import numpy as np
from heapq import heappush, heappop
import math
import random

class BorderlineSmote():
	k = 10

	def dist(v1, v2, n):
		sum = 0
		for i in range(n):
			sum += math.pow(v1[i]-v2[i], 2)
		return sum

	def kNearestNeighbours(mat, r, c, cur):
		heapq = []
		for i in range(r):
			dis = BorderlineSmote.dist(mat[i], mat[cur], c)
			if dis == 0:
				continue
			flag = False
			for j in range(len(heapq)):
				if (mat[heapq[j][1]] == mat[i]).all():
					flag = True
					break
			if flag:
				continue
			if len(heapq) < BorderlineSmote.k:
				heappush(heapq, (-dis, i))
			elif dis < -heapq[0][0]:
				heappop(heapq)
				heappush(heapq, (-dis, i))
		vec = np.zeros((BorderlineSmote.k), dtype=int)
		for i in range(BorderlineSmote.k):
			vec[i] = heapq[i][1]
		return vec

	def pupulate(ret, mat, num, cur, vec, N, c):
		for i in range(N):
			idx = vec[random.randint(0, BorderlineSmote.k-1)]
			for j in range(c):
				fac = random.random()
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

		for i in range(T):
			if i % 20 == 0:
				print('%d of %d (in finding danger set...)' % (i, T))
			vec = BorderlineSmote.kNearestNeighbours(matall, matall.shape[0], n, i)
			typ = BorderlineSmote.sampleType(vec, matall, vecall)
			if typ == 1:
				choice[danger] = i
				danger += 1

		N = int(tar / danger)
		tar = N * danger
		ret = np.zeros((tar, n))
		tot = 0		

		for i in range(danger):
			vec = BorderlineSmote.kNearestNeighbours(mat, T, n, choice[i])
			BorderlineSmote.pupulate(ret, mat, tot, i, vec, N, n)
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