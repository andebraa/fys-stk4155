from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import functions as f



def learning_rate(t, t0 = 3, t1 = 50):
    return t0/(t+t1)

def f_test(x, y):
    return 5 + 2*x + 3*x**2

def sgdm(m, degrees, n_epochs, b, eta, noise=0, gamma = 0): #stocastic gradient decent with momentum
    np.random.seed(1337)
    x = np.random.rand(m,degrees) #+1?
    y = np.random.rand(m,degrees) #+1?

    X_mesh, Y_mesh = np.meshgrid(x, y)

    z = f.FrankeFunction(X_mesh, Y_mesh) + noise*np.random.randn(X_mesh.shape[0], Y_mesh.shape[0])

    z= np.ravel(z)
    X = f.X_make(X_mesh,Y_mesh, degrees)

    #SPLIT AND SCALE
    X_tr, X_te, z_tr, z_te = train_test_split(X,z, test_size=0.3)
    scaler = StandardScaler()
    # X_tr = scaler.fit(X_tr).transform(X_tr)
    # z_tr = scaler.transform(z_tr.reshape(-1,1))
    # z_te = scaler.fit(z_te).transform(z_te)

                                                # removes the mean and scales each feature/variable to unit variance
    scaler.fit(X_tr)                         # compute the mean and std to be used for later scaling
    X_tr= scaler.transform(X_tr)  # perform standardization by centering and scaling
    X_te = scaler.transform(X_te)    # fit to data, then transform it
    z_tr = z_tr.reshape(-1,1)
    z_te = z_te.reshape(-1,1)
    scaler.fit(z_tr)
    z_tr = scaler.transform(z_tr)
    z_te = scaler.transform(z_te)

    l = int((degrees+1)*(degrees+2)/2) #length of design matrix row
    beta = np.random.randn(l,1) #length of a design matrix row
    #b = int(m/batch_num) #batch size
    batch_num = int(m/b)
    if m%batch_num:
        print('warning; batch number and dataset not compatible')

    v = 0
    mse_eval = np.zeros(n_epochs)
    index_array = np.arange(m) #indexes of rows
    batch_array= np.arange(batch_num)
    batch_array *=b
    for epoch in range(n_epochs):
        np.random.shuffle(index_array)
        for i in range(batch_num): #m is number of batches
            xi = X_tr[index_array[batch_array[i]]: (index_array[(batch_array[i]+1)])]
            zi = z_tr[index_array[batch_array[i]]: (index_array[(batch_array[i]+1)])]

            gradients = 2/b * xi.T @ (xi @ beta - zi.reshape(-1,1)) #derived from cost function
            #eta = 0.001#learning_rate(epoch*m+i)
            v = gamma*v + eta*gradients
            beta = beta - v
        z_eval = X_te.dot(beta)
        mse_eval[epoch] = f.MSE(z_te, z_eval)
    beta_ols = f.OLS(X_tr, z_tr)
    z_ols = X_te.dot(beta_ols)
    mse_beta = f.MSE(z_te, z_ols)
    return beta, mse_eval, mse_beta
print("beta from own sdg")
n_epochs = 1500
etas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
for i in etas:
    beta, mse_eval, mse_beta = sgdm(\
    m = 60, degrees = 4, n_epochs=n_epochs, b=5, eta = i, gamma = 0.5)
    plt.plot(np.arange(n_epochs), mse_eval, label=f"eta: {i}")

plt.plot(np.arange(n_epochs), mse_beta*np.ones(n_epochs))
plt.ylim(-10, 1E3)
plt.legend()
plt.show()
print(beta)
