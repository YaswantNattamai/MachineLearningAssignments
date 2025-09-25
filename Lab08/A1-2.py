# A1
import numpy as np


# Summation unit (weighted sum including bias)
def summation_unit(x, w):
    """x: 1D array of inputs (including bias input 1), w: weight array same length"""
    x = np.asarray(x)
    w = np.asarray(w)
    return np.dot(w, x)


# Activation functions
def step_activation(net):
    return 1.0 if net >= 0.0 else 0.0


def bipolar_step_activation(net):
    return 1.0 if net >= 0.0 else -1.0


def sigmoid_activation(net):
    return 1.0 / (1.0 + np.exp(-net))


def tanh_activation(net):
    return np.tanh(net)


def relu_activation(net):
    return net if net > 0 else 0.0


def leaky_relu_activation(net, alpha=0.01):
    return net if net > 0 else alpha * net


# Comparator / error (sum of squared errors across samples)
def sum_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 0.5 * np.sum((y_true - y_pred) ** 2)


# Helper to apply chosen activation elementwise (for vector inputs)
ACTIVATIONS = {
'step': step_activation,
'bipolar': bipolar_step_activation,
'sigmoid': sigmoid_activation,
'tanh': tanh_activation,
'relu': relu_activation,
'leaky_relu': lambda net: leaky_relu_activation(net, alpha=0.01)
}

# A2 PERCEPTRON TRAINING
import matplotlib.pyplot as plt


def perceptron_train(X, T, w_init, lr=0.05, activation='step', max_epochs=1000, tol=0.002):
    """X: shape (N, n_inputs) WITHOUT bias column; T: targets (N,) ; w_init: array length n_inputs+1 (bias first)
    returns trained weights, list of SSE per epoch, epochs run
    """
    X = np.asarray(X)
    N, n = X.shape
    # Add bias input 1 as first column
    Xb = np.hstack([np.ones((N,1)), X])
    w = np.array(w_init, dtype=float).copy()
    sse_list = []
    act_fn = ACTIVATIONS[activation]


    for epoch in range(1, max_epochs+1):
        outputs = []
        for i in range(N):
            net = summation_unit(Xb[i], w)
            y = act_fn(net)
            outputs.append(y)
            # perceptron delta rule (target - output)
            delta = (T[i] - y)
            # weight update: w = w + lr * delta * x
            w = w + lr * delta * Xb[i]
        sse = sum_squared_error(T, outputs)
        sse_list.append(sse)
        if sse <= tol:
            break
    return w, sse_list, epoch


# AND dataset
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
T_and = np.array([0,0,0,1])
w_init = np.array([10.0, 0.2, -0.75])


trained_w, sse_list, epochs_ran = perceptron_train(X_and, T_and, w_init, lr=0.05, activation='step')
print('Trained weights:', trained_w)
print('Epochs:', epochs_ran)


# Plot epochs vs SSE
plt.figure()
plt.plot(np.arange(1, len(sse_list)+1), sse_list)
plt.xlabel('Epoch')
plt.ylabel('Sum-squared-error')
plt.title('Perceptron training (AND) - Step activation')
plt.grid(True)
plt.show()

