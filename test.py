import numpy as np
from normalize import Normalize

class Test():
	m = 538
	n = 11
	matx = np.zeros((m, n))
	label = np.zeros((m), dtype=int)
	labelp = np.zeros((m), dtype=int)

	def __init__(self):
		self.matx[:,10] = 1
		self.input()

	def input(self):
		fin = open('F:\\data\\ml\\2\\page_blocks_test_feature.txt', 'r')
		lines = fin.readlines()
		row = 0
		for line in lines:
			list = line.strip('\n').split(' ')
			self.matx[row][0:10] = list
			row += 1
		Normalize.normalize(self.matx)

		fin = open('F:\\data\\ml\\2\\page_blocks_test_label.txt', 'r')
		lines = fin.readlines()
		row = 0
		for line in lines:
			list = line.strip('\n')
			self.label[row] = list[0]
			row += 1

	def predict(self, beta, k):
		ans = np.zeros((k))
		for i in range(self.m):
			for type in range(k):
				ans[type] = np.dot(self.matx[i], beta[type])
			self.labelp[i] = np.argmax(ans)+1

		rel = np.zeros((k, 2, 2), dtype = int)
		for i in range(self.m):
			if (self.label[i] == self.labelp[i]):
				for j in range(k):
					if j+1 == self.label[i]:
						rel[j][0][0] += 1
					else:
						rel[j][1][1] += 1
			else:
				for j in range(k):
					if j+1 == self.label[i]:
						rel[j][0][1] += 1
					elif j+1 == self.labelp[i]:
						rel[j][1][0] += 1
					else:
						rel[j][1][1] += 1

		for i in range(k):
			print('class %d' % (i+1))
			print(rel[i])
			Precision = rel[i][0][0] / (rel[i][0][0] + rel[i][1][0])
			Recall = rel[i][0][0] / (rel[i][0][0] + rel[i][0][1])
			print('Precision: %.3f%%' % (Precision*100))
			print('Recall: %.3f%%' % (Recall*100))
