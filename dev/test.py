import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10)
plt.subplot(221)
plt.plot(x, np.sin(x), c=np.random.rand(3,1), ls='--')
plt.grid(True)
plt.subplot(221)
plt.plot(x, np.sin(x)+1, c=np.random.rand(3,1), lw=4.0)
plt.title("asdf")
plt.grid(True)
plt.subplot(222)
plt.plot(x, np.sin(x)+2, c=np.random.rand(3,1), ls='--')
plt.grid(True)
plt.subplot(222)
plt.plot(x, np.sin(x)+3, c=np.random.rand(3,1), ls='--')
plt.grid(True)
plt.show()
