import numpy as np
import math

class Normalize():
	def normalize(mat):
		m = mat.shape[0]
		n = mat.shape[1]
		for i in range(n):
			mean = np.mean(mat[:,i])
			std = math.sqrt(np.var(mat[:,i]))
			if std == 0:
				continue
			for j in range(m):
				mat[j][i] = (mat[j][i] - mean) / std