import random
import numpy as np

class EasyEnsemble():
	def genNew(matp, matn, c):
		mat = np.vstack((matp, matn))

		nump = matp.shape[0]
		numn = matn.shape[0]

		if nump > numn:
			mat = np.zeros((numn, c))			
			for i in range(numn):
				mat[i] = matp[random.randint(0, nump-1)]
			mat = np.vstack((mat, matn))
			vec = np.hstack((np.ones((numn)), np.zeros((numn))))
		else:
			mat = np.zeros((nump, c))			
			for i in range(nump):
				mat[i] = matn[random.randint(0, numn-1)]
			mat = np.vstack((matp, mat))
			vec = np.hstack((np.ones((nump)), np.zeros((nump))))

		return (mat, vec)