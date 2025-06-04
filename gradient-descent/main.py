import copy, math
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[2104, 5, 1, 45], [1416, 3 , 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print("x_train:", x_train)
print("y_train:", y_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict(x, b, w):
    p = np.dot(x, w) + b
    return p

x_vec = x_train[0, :] # row vector from data set
f_wb = predict(x_vec, b_init, w_init)
print("f_wb:", f_wb)

# make initial prediction
# calclate cost function

def cost_function(x, y, b, w):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost += (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

cost = cost_function(x_train, y_train, b_init, w_init)
print("cost:", cost)