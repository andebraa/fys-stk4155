"""
Python file containing all the activation functions used in Neuralnet
and frankefunc_call .py
"""
import numpy as np

class sigmoid:
	def __call__(self,z):
		return(1/(1 + np.exp(-z)))
	def deriv(self,z):
		sig = self.__call__(z)
		return sig-sig**2 #https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e

class identity:
	def __call__(self, z):
		return z
	def deriv(self, z):
		return np.ones(z.shape)
