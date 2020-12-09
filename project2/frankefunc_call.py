import numpy as np
from NeuralNet import BilekFlowDNN, BilekFlowDenseLayer, MSE, sigmoid


def FrankeFunction_call(m, degrees, n_epochs, b, eta, noise=0, gamma = 0): #stocastic gradient decent with momentum
    layer1 = BilekFlowDenseLayer
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
