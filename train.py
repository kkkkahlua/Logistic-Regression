import numpy as np
import math

class Train():
	m = 4935
	n = 11
	EPS = 0.0001
	matx = np.zeros((m, n))
	label = np.zeros((m), dtype=int)

	def __init__(self):

		self.matx[:,10] = 1
		pass

	def input(self):
		fin = open('F:\\data\\ml\\2\\page_blocks_train_feature.txt', 'r')
		lines = fin.readlines()
		row = 0
		for line in lines:
			list = line.strip('\n').split(' ')
			self.matx[row][0:10] = list
			row += 1
		fin = open('F:\\data\\ml\\2\\page_blocks_train_label.txt', 'r')
		lines = fin.readlines()
		row = 0
		for line in lines:
			list = line.strip('\n')
			self.label[row] = list[0]
			row += 1

	def calc_p1(self, x, beta):
		print(np.dot(x, beta))
		temp = math.exp(np.dot(x, beta))
		return temp / (1+temp)

	def calc_deri_1(self, beta):
		vec = np.zeros((self.n))
		for i in range(self.m):
			#print(self.calc_p1(self.matx[i], beta))
			vec += self.matx[i] * (self.calc_p1(self.matx[i], beta) - self.label[i])
		return vec

	def calc_deri_2(self, beta):
		mat = np.zeros((self.n, self.n))
		c = np.zeros((self.n, 1))
		r = np.zeros((1, self.n))
		for i in range(self.m):
			c[:,0] = self.matx[i]
			r[0:] = self.matx[i]
			p1 = self.calc_p1(self.matx[i], beta)
			mat += np.dot(c, r) * p1 * (1-p1)
		return mat

	def close(self, x, y):
		for i in range(self.n):
			if (abs(x[i]-y[i]) > self.EPS):
				return False
		return True

	def recur(self, beta):
		ret = beta - np.dot(self.calc_deri_2(beta).T, self.calc_deri_1(beta))
		if (self.close(beta, ret)):
			return ret
		else:
			return self.recur(ret)

	def solve(self):
		self.input()

		'''
		for x in range(50):
			print(self.matx[x])
			print(self.label[x])
		'''

		beta = np.zeros((self.n))
	#	beta = self.calc_deri_1(beta)
	#	print(beta)
	#	mat = self.calc_deri_2(beta)
	#	print(mat)
		print(self.recur(beta))