from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from functions import OLS, Ridge_func, MSE, FrankeFunction, X_make
def CV(max_deg, fold_num, noise, method):
    """
    applies cross validation for a given x and y arrays (must be same size)
    initially splits into train and test data,
    then loops over given amount of folds and finds MSE, for each degree ,
    With the lowest MSE found we use this on the initial test data from the
    train test split on x and y to validate.

    args:
        x, y (np.array): initial datapoints
        max_deg (int): highest degree, (goes from 0 to this values
        fold_num (int) : number of k-folds performed
    returns:
        MSE averaged over fold_num number of iterations
    """

    np.random.seed(130)
    scaler = StandardScaler()
    x = np.sort(np.random.uniform(0, 1, dp))
    y = np.sort(np.random.uniform(0, 1, dp))
    #x,y = np.meshgrid(x,y)

    X_tr_, X_te_, Y_tr_, Y_te_ = train_test_split(x,y, test_size=0.2)

    X_tr, Y_tr = np.meshgrid(X_tr_, Y_tr_)
    X_te, Y_te = np.meshgrid(X_te_, Y_te_)

    z_tr = np.ravel(FrankeFunction(X_tr, Y_tr)+ noise*np.random.randn( X_tr.shape[0], X_tr.shape[0]))
    z_te = np.ravel(FrankeFunction(X_te, Y_te) + noise*np.random.randn(X_te.shape[0], X_te.shape[0]))
    MSE_train_values = np.zeros(max_deg)
    MSE_test_values = np.zeros(max_deg)

    print(len(z_tr))
    for k, deg in enumerate(range(0,max_deg)):
        #Degrees loop that contains K-fold


        X_design = X_make(X_tr, Y_tr, deg)
        X_design[:,0] = 1

        X_master = np.array(np.array_split(X_design, fold_num))
        z_tr_spl = np.array(np.array_split(z_tr, fold_num)) #This nasty cunt has to be dividable by fold_num in order to split it
        X_train_fold = np.zeros((X_master.shape[0]-1, X_master.shape[1], X_master.shape[2]))
        z_train_fold = np.zeros((len(z_tr) - int(len(z_tr)/fold_num)))#note; this cunt is scaled and split training data

        MSEtrain = np.zeros(fold_num)
        MSEtest = np.zeros(fold_num)
        test = np.empty(3)
        print(test)
        for i in range(fold_num):

            X_test_fold = X_master[i]
            z_test_fold = z_tr_spl[i]

            for ex in range(fold_num):
                if ex == i:
                    pass
                else:
                    X_train_fold = X_master[ex,:,:]
                    z_train_fold = z_tr_spl[ex,:]

            #Scaling
            X_train_fold_s = scaler.fit(X_train_fold).transform(X_train_fold) #NOTE remember to scale the rest of the data!!
            #since we are using part of the original train data to test each fold,
            #we must make sure to not scale this part of the data, and instead
            #fit the given training data for each fold

            if method == OLS:
                z_test_fold_hat = X_test_fold.dot(method(X_train_fold_s, z_train_fold))
                z_train_fold_hat = X_train_fold_s.dot(method(X_train_fold_s, z_train_fold))
            elif method == Ridge_func:
                z_test_fold_hat = X_test_fold.dot(method(X_train_fold_s, z_train_fold))
                z_train_fold_hat = X_train_fold_s.dot(method(X_train_fold_s, z_train_fold))
            elif method == 'lasso':
                clf_lasso = skl.Lasso(alpha=lmda).fit(X_train_fold_s)
                z_test_fold_hat = clf_lasso.predict(X_test_fold)


            MSEtrain[i] = MSE(z_test_fold, z_test_fold_hat)
            MSEtest[i] = MSE(z_train_fold, z_train_fold_hat)
        MSE_train_values[k] = np.mean(MSEtrain)
        MSE_test_values[k] = np.mean(MSEtest)
    return MSE_train_values, MSE_test_values

max_deg = 10
folds = 5
dp = 50
MSE_train_values, MSE_test_values = CV(max_deg, folds, 0.1, OLS)

plot_degrees = np.arange(0, max_deg, 1)

plt.plot(plot_degrees, MSE_train_values, label="train")
plt.plot(plot_degrees, MSE_test_values, '--', label="test")
plt.legend()
plt.show()
