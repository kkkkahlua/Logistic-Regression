import numpy as np
import math
from OvR import OvR
from smote import Smote
from newton import Newton
from descent import Descent
from gradascent import GradAscent
from dampednewton import DampedNewton
from logistic import Logistic
from normalize import Normalize

class Train():
	m = 4935
	n = 11

	matx = np.zeros((m, n))
	label = np.zeros((m), dtype=int)

	pos = 0

	def __init__(self):
		self.matx[:,10] = 1
		self.input()

	def input(self):
		fin = open('F:\\data\\ml\\2\\page_blocks_train_feature.txt', 'r')
		lines = fin.readlines()
		row = 0
		for line in lines:
			list = line.strip('\n').split(' ')
			self.matx[row][0:10] = list
			row += 1
		Normalize.normalize(self.matx)

		fin = open('F:\\data\\ml\\2\\page_blocks_train_label.txt', 'r')
		lines = fin.readlines()
		row = 0
		for line in lines:
			list = line.strip('\n')
			self.label[row] = list[0]
			row += 1


	def solve(self, pos):
		self.pos = pos

		ret1 = OvR.separate(self.matx, self.label, pos, self.m, self.n)

		ret2 = Smote.genNew(ret1[0], ret1[1], self.n)

		mat = ret2[0]
		vec = ret2[1]
	
	#	beta = Descent.solve(mat, vec)

		beta = DampedNewton.solve(mat, vec)

	#	beta = Newton.solve(mat, vec)

	#	beta = DampedNewton.solve(mat, vec)
		'''
		vecm = np.zeros((vec.shape[0], 1))
		vecm[:,0] = vec
		logistic = Logistic(mat, vecm)
		beta = logistic.Newton()

		print('beta', beta)

		#beta = GradAscent.solve(mat, vec)
		'''
		return beta