import numpy as np

class OvR():
	def separate(mat, vec, tar, sizer, sizec):
		matp = np.zeros((sizer, sizec))
		matn = np.zeros((sizer, sizec))
		nump = numn = 0
		for i,x in enumerate(vec):
			if x == tar:
				matp[nump] = mat[i]
				nump += 1
			else:
				matn[numn] = mat[i]
				numn += 1
		matp = matp[0:nump]
		matn = matn[0:numn]
		return (matp, matn)