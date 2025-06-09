import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/ (1 + np.exp(-z))

z_data = np.arange(-10, 11)
y = sigmoid(z_data)
np.set_printoptions(precision=3, suppress=True)
print(np.c_[z_data, y])

fig, ax = plt.subplots()
ax.plot(z_data, y, label='sigmoid', color='blue')
ax.set_xlabel('z')
ax.set_ylabel('sigmoid(z)')
ax.set_title('Sigmoid Function')
ax.legend()
plt.grid()
plt.show()

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0
plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)