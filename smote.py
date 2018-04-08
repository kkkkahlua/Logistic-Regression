import numpy as np
from heapq import heappush, heappop
import math
import random

class Smote():
	k = 10

	def dist(v1, v2, n):
		sum = 0
		for i in range(n):
			sum += math.pow(v1[i]-v2[i], 2)
		return sum

	def kNearestNeighbours(mat, r, c, cur):
		heapq = []
		for i in range(r):
			dis = Smote.dist(mat[i], mat[cur], c)
			if dis == 0:
				continue
			flag = False
			for j in range(len(heapq)):
				if (mat[heapq[j][1]] == mat[i]).all():
					flag = True
					break
			if flag:
				continue
			if len(heapq) < Smote.k:
				heappush(heapq, (-dis, i))
			elif dis < -heapq[0][0]:
				heappop(heapq)
				heappush(heapq, (-dis, i))
		vec = np.zeros((Smote.k), dtype=int)
		for i in range(Smote.k):
			vec[i] = heapq[i][1]
		return vec

	def pupulate(ret, mat, num, cur, vec, N, c):
		for i in range(N):
			idx = vec[random.randint(0, Smote.k-1)]
			for j in range(c):
				fac = random.random()
				ret[num+i][j] = mat[cur][j] + fac * (mat[idx][j] - mat[cur][j])


	def overSample(mat, T, N, n):
		if N == 0:
			return mat

		ret = np.zeros(((T*N), n))
		tot = 0

		for i in range(T):
			vec = Smote.kNearestNeighbours(mat, T, n, i)
			Smote.pupulate(ret, mat, tot, i, vec, N, n)	
			tot += N

		return ret

	def genNew(matp, matn, c):
		mat = np.vstack((matp, matn))

		nump = matp.shape[0]
		numn = matn.shape[0]
		print(nump)
		print(numn)
		vec = np.hstack((np.ones((nump), dtype=int), np.zeros((numn), dtype=int)))
		
		if (nump > numn):
			ret = Smote.overSample(matn, numn, int(nump/(2*numn))-1, c)
			num = ret.shape[0]
			mat = np.vstack((mat, ret))
			vec = np.hstack((vec, np.zeros((num), dtype=int)))
		elif (nump < numn):
			ret = Smote.overSample(matp, nump, int(numn/(2*nump))-1, c)
			num = ret.shape[0]
			mat = np.vstack((mat, ret))
			vec = np.hstack((vec, np.ones((num), dtype=int)))
		'''
		fout = open('F:\\data\\ml\\2\\train_4.txt', 'w')
		for i in range(mat.shape[0]):
			fout.write(str(mat[i]))
			fout.write('\n')
		for i in range(mat.shape[0]):
			fout.write(str(vec[i]))
			fout.write('\n')
		'''
		return (mat, vec)