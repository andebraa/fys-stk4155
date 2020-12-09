from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import functions as f
import numpy as np

def bootstrap(x,y, max_deg, boots_num):
    np.random.seed(130)
    """
    applies the bootstrap algorithm

    args:
        x,y (np.array): initial datapoints
        max_deg (int):
        boots_num (int): number of bootstraps
    """

    x,y = np.meshgrid(x,y)
    z = np.ravel(f.FrankeFunction(x, y)+ 0.5*np.random.randn(np.shape(x)[0], np.shape(y)[1]))

    MSE_degree_values = np.zeros(max_deg)
    MSE_test_degree_values = np.zeros(max_deg)
    MSE_train_values = np.zeros(boots_num)
    MSE_test_values = np.zeros(boots_num)
    for k,deg in enumerate(range(1,max_deg)):
        #Degrees loop that contains K-fold
        X_design = f.X_make(x, y, deg)
        scaler = StandardScaler()

        X_tr, X_te, z_tr, z_te = train_test_split(X_design, z, test_size=0.2)
        scaler.fit(X_tr)

        X_train = scaler.transform(X_tr)
        X_test = scaler.transform(X_te)
        #doing this AFTER train test split. otherwise the test data
        #gets affected by the train data
        z_bootstrap = np.empty(int(len(z_tr)))
        index_array = np.arange(0,len(z_tr),1)


        for i in range(boots_num):
            indx = resample(index_array, random_state = 0)
            z_bootstrap = z_tr[indx]

            z_test = X_test.dot(f.OLS(X_train[indx,:], z_bootstrap))
            z_train = X_train.dot(f.OLS(X_train[indx,:], z_bootstrap))
            MSE_train_values[i] = f.MSE(z_tr, z_train)
            MSE_test_values[i] = f.MSE(z_te, z_test)

        MSE_degree_values[k] = np.sum(MSE_train_values)/boots_num
        MSE_test_degree_values[k] = np.sum(MSE_test_values)/boots_num
    return MSE_degree_values, MSE_test_degree_values

max_deg = 20
bootstraps = 10
dp = 50
x = np.linspace(0, 1, dp)
y = np.linspace(0, 1, dp)

MSE_degree_values, MSE_test_degree_values = bootstrap(x,y, max_deg, bootstraps)
plot_degrees = np.arange(0,max_deg, 1)
plt.plot(plot_degrees[:-1], MSE_degree_values[:-1])
plt.plot(plot_degrees[:-1], MSE_test_degree_values[:-1], '--', label="test on test")
plt.show()
