import numpy as np
import matplotlib.pyplot as plt


x = np.arange(10)
y = np.sin(x)
for i in range(3):
    plt.plot(x,y*i,label=f"{i}")
plt.legend()
plt.show()
