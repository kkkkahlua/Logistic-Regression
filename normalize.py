import numpy as np

class Normalize():
	def normalize(mat):
		m = mat.shape[0]
		n = mat.shape[1]
		for i in range(n):
			E = np.mean(mat[:,i])
			D = np.var(mat[:,i])
			if D == 0:
				continue
			for j in range(m):
				mat[j][i] = (mat[j][i] - E) / D