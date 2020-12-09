"""
Python file containing all the cost funcitons used in Neuralnet
and frankefunc_call .py 
"""


import numpy as np

class MSE:
	def __call__(self,y_pred,y):
		return(np.mean((y_pred - y)**2))

	def deriv(self,y_pred,y):
		return(2*(y_pred - y)/y.shape[0])
