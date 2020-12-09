import numpy as np
import functions as f
#ACTIVATION FUNCTIONS
class sigmoid:
	def __call__(self,z):
		return(1/(1 + np.exp(-z)))
	def deriv(self,z):
		sig = self.__call__(z)
		return sig-sig**2 #https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e

#COST FUNCTIONS
class MSE:
	def __call__(self,y_pred,y):
		return(np.mean((y_pred - y)**2))

	def deriv(self,y_pred,y):
		return(2*(y_pred - y)/y.shape[0])

#LAYER DEFINITIONS
class BilekFlowDenseLayer:
	def __init__(self,inputs,outputs,activation):
		"""
		Typically weights are initialized with small values distributed around
		zero, drawn from a uniform or normal distribution.
		Setting all weights to zero means all neurons give the same output,
		making the network useless.
		"""
		self.inputs, self.outputs, self.activation  = inputs, outputs, activation
		self.w = 0.01*np.random.randn(inputs, outputs) #only initially
		self.b = 0.01*np.random.randn(1,outputs) #or


	def __call__(self,x):

		self.z = x@self.w + self.b #self.w@x.T + self.b
		print(np.shape(self.z))
		self.a = self.activation()(self.z) #ex sigmoid, MSE
		self.a_deriv = self.activation().deriv(self.z)
		print(np.shape(self.a_deriv))
		print('heil')
		return(self.a)

#DENSE NEURAL NETWORKS
class BilekFlowDNN:
	def __init__(self,layers,cost):
		self.layers = layers
		self.cost = cost

	def __call__(self,x):
		for layer in self.layers:
			x = layer(x)
		return(x)

	def lr(self, t, t0=3, t1=50):
		return t0/(t+t1)

	def backprop(self,x,y, eta):
		y_pred = self.__call__(x)
		aL = self.layers[-1].a_deriv #activation of layer L
		print('aL')
		print(np.shape(aL))
		dCda = self.cost().deriv(y_pred,y)
		print('heill')
		print(np.shape(dCda))
		delta_L = aL*dCda
		for i in reversed(range(1, len(self.layers)-1)):
			delta_l = (delta_L@self.layers[i+1].w.T)*layers[i].a_deriv
			print('cuuunt')
			print(np.shape(layers[i].w))
			print(np.shape(delta_l))
			print(np.shape(self.layers[i-1].a))
			layers[i].w = layers[i].w - eta*delta_l*self.layers[i-1].a #introduce eta
			layers[i].b = layers[i].b - eta*delta_l
			delta_L = delta_l
		delta_l = (delta_L@self.layers[1].w.T)*layers[0].a_deriv
		#delta_l = delta_L@layers[0].a_deriv*self.layers[1].w
		layers[0].w = layers[0].w - eta * delta_l @ x.T

	def __str___(self):
		return layers[-1].a





if __name__ == '__main__':
	layer1 = BilekFlowDenseLayer(2,10, sigmoid)
	layer2 = BilekFlowDenseLayer(10,10, sigmoid)
	layer3 = BilekFlowDenseLayer(10,1, sigmoid)
	layers = [layer1,layer2, layer3]
	m = 100
	eta= 0.001
	X_tr, z_tr, X_te, z_te = f.make_franke_data(100)
	a = BilekFlowDNN(layers,MSE)
	a.backprop(X_tr, z_tr, eta)
	print(layer3.a)
