from train import Train
import numpy as np

train = Train()

for i in range(5):
	beta = train.solve(i+1)
