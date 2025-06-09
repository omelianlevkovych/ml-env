import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """Compute the sigmoid of z."""
    return 1 / (1 + np.exp(-z))


def plt_one_addpt_onclick(x, y, w, b, logistic=False):
    """Plot data and model line and allow adding one point via mouse click."""
    fig, ax = plt.subplots()

    ax.scatter(x, y, marker="x", c="red", label="data")
    x_model = np.linspace(x.min() - 0.5, x.max() + 0.5, 50)
    if logistic:
        # w may be a one-element array; use elementwise multiplication
        y_model = sigmoid(w * x_model + b)
        ax.set_ylim(-0.1, 1.1)
    else:
        y_model = np.dot(x_model, w) + b
    ax.plot(x_model, y_model, color="blue", label="model")
    ax.legend()

    added_pt = []

    def _onclick(event):
        if event.inaxes != ax:
            return
        added_pt.append((event.xdata, event.ydata))
        ax.scatter(event.xdata, event.ydata, color="green", marker="o", label="added")
        fig.canvas.draw()
        fig.canvas.mpl_disconnect(cid)

    cid = fig.canvas.mpl_connect("button_press_event", _onclick)
    plt.show()

    return added_pt[0] if added_pt else None

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
