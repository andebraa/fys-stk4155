import numpy as np
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def X_make(x_, y_, n):
    x = np.ravel(x_)
    y = np.ravel(y_)
    N = len(x)
    global l
    l = int((n+1)*(n+2)/2)
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k) #binominal theorem
    return X
def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def OLS(X, y):
    betta = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(y)
    return betta

def Ridge_func(X_, z_, lamda):
    #ridge method
    l = int((n+1)*(n+2)/2)
    beta = np.linalg.inv(X_.T@X_ + np.identity(l)*lamda).dot(X_.T@z_)

    return beta

def MSE(y, y_):
    #return np.mean(np.mean(y - y_)**2, axis=1, keepdims=True) #Elins bootstrap method
    return np.mean(np.square(y-y_))

def R_square(y, y_):
    y_mean = np.mean(y)
    a = np.sum(np.square(y - y_))
    b = np.sum(np.square(y-y_mean))
    return 1- (a/b)

def make_franke_data(m, noise = 0):
    np.random.seed(1337)
    x = np.random.rand(m) #+1?
    y = np.random.rand(m) #+1?

    X_mesh, Y_mesh = np.meshgrid(x, y)

    z = FrankeFunction(X_mesh, Y_mesh) + noise*np.random.randn(X_mesh.shape[0], Y_mesh.shape[0])

    z= np.ravel(z)
    X = X_make(X_mesh, Y_mesh, 1)
    X = X[:,1:]

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

    return X_tr, z_tr, X_te, z_te
