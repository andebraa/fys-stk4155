from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import functions as f
import numpy as np
import sys

"""
Module made for applying linear regression in three dimensions on the Frankes function.
Code is made for project 1 in fys-stk4155 at UiO.
"""

np.random.seed(1488)
max_degree = 14
dp = 20
deg = np.arange(0,max_degree,1)
MSEtest = np.zeros(max_degree)
MSEtrain = np.zeros(max_degree)

for i, degree in enumerate(deg):
    # Make data.
    x = np.linspace(0, 1, dp)
    y = np.linspace(0, 1, dp)
    x, y = np.meshgrid(x, y)

    X = f.X_make(x, y, degree)

    z = f.FrankeFunction(x, y)+ 0.1*np.random.randn(dp,dp)
    z = np.ravel(z)

    #seperate and scale
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    print(f"z: {np.shape(z)}")
    scaler = StandardScaler()
    scaler.fit(X)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f" z_train: {np.shape(z_train)}")
    print(f" X: {np.shape(X)}")
    print(f" X_train: {np.shape(X_train)}")
    z_test_scaled = X_test_scaled.dot(f.Ridge_func(X_train_scaled, z_train, 1E-5))
    z_train_scaled = X_train_scaled.dot(f.Ridge_func(X_train_scaled, z_train, 1E-5))
    print(f"x: {np.shape(x)}")
    print(f"y: {np.shape(y)}")

    MSEtrain[i] = f.MSE(z_train, z_train_scaled)
    MSEtest[i] = f.MSE(z_test, z_test_scaled)


plt.plot(deg, MSEtest, label="test")
plt.plot(deg, MSEtrain, label="train")
plt.legend()
plt.show()
"""
#optional plotting of surface
z_plot = np.reshape(z_, (25000, 25000))
#print(np.shape(z_))
# Plot the surface.

surf = ax.plot_surface(x, y, z_plot, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
"""
