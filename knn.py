from heapq import heappush, heappop
import numpy as np
import math

class KNN():
	k = 10

	def dist(v1, v2, n):
		sum = 0
		for i in range(n):
			sum += math.pow(v1[i]-v2[i], 2)
		return sum

	def kNearestNeighbours(mat, r, c, cur):
		heapq = []
		for i in range(r):
			dis = KNN.dist(mat[i], mat[cur], c)
			if dis == 0:
				continue
			flag = False
			for j in range(len(heapq)):
				if (mat[heapq[j][1]] == mat[i]).all():
					flag = True
					break
			if flag:
				continue
			if len(heapq) < KNN.k:
				heappush(heapq, (-dis, i))
			elif dis < -heapq[0][0]:
				heappop(heapq)
				heappush(heapq, (-dis, i))
		vec = np.zeros((KNN.k), dtype=int)
		for i in range(KNN.k):
			vec[i] = heapq[i][1]
		return vec