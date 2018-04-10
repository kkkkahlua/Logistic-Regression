import numpy as np
import math
from OvR import OvR
from smote import Smote
from descent import Descent
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

		mat, vec = Smote.genNew(ret1[0], ret1[1], self.n)
	
		beta = Descent.solve(mat, vec)

		print('beta', beta)

		return beta
		