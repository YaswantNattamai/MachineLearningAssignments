# A10: Two-output AND gate MLP in same style as A1
import numpy as np
import matplotlib.pyplot as plt

# Summation unit
def summation_unit(x, w):
    x = np.asarray(x)
    w = np.asarray(w)
    return np.dot(w, x)

# Sigmoid activation
def sigmoid_activation(net):
    return 1.0 / (1.0 + np.exp(-net))

# SSE across samples
def sum_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 0.5 * np.sum((y_true - y_pred) ** 2)

# Forward pass for two-layer MLP
def forward_pass(X, W1, W2):
    N = X.shape[0]
    Xb = np.hstack([np.ones((N,1)), X])
    out_h = sigmoid_activation(Xb.dot(W1))
    out_hb = np.hstack([np.ones((N,1)), out_h])
    out_o = sigmoid_activation(out_hb.dot(W2))
    return out_o

# Train MLP for two-output AND
def mlp_train_AND_2out(X, T, hidden_neurons=2, lr=0.1, max_epochs=5000, tol=0.002, random_seed=3):
    X = np.asarray(X)
    T = np.asarray(T)
    N, n_input = X.shape
    n_output = T.shape[1]

    rng = np.random.default_rng(random_seed)
    W1 = rng.normal(scale=0.5, size=(n_input+1, hidden_neurons))
    W2 = rng.normal(scale=0.5, size=(hidden_neurons+1, n_output))

    sse_list = []

    for epoch in range(1, max_epochs+1):
        # Forward
        Xb = np.hstack([np.ones((N,1)), X])
        out_h = sigmoid_activation(Xb.dot(W1))
        out_hb = np.hstack([np.ones((N,1)), out_h])
        out_o = sigmoid_activation(out_hb.dot(W2))

        # Compute error
        err = T - out_o
        sse = 0.5 * np.sum(err**2)
        sse_list.append(sse)

        if sse <= tol:
            break

        # Backpropagation
        delta_o = err * out_o * (1 - out_o)
        grad_W2 = out_hb.T.dot(delta_o)
        delta_h = (delta_o.dot(W2[1:].T)) * out_h * (1 - out_h)
        grad_W1 = Xb.T.dot(delta_h)

        # Update weights
        W2 += lr * grad_W2
        W1 += lr * grad_W1

    return W1, W2, sse_list, epoch

# AND 2-output dataset
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
T_and_2out = np.array([[1,0],[1,0],[1,0],[0,1]])

# Train
W1_two, W2_two, sse_list, epochs_ran = mlp_train_AND_2out(X_and, T_and_2out)

# Forward pass to get outputs
outputs = forward_pass(X_and, W1_two, W2_two)
pred_classes = (outputs >= 0.5).astype(int)

print("Trained weights W1:\n", W1_two)
print("Trained weights W2:\n", W2_two)
print("Epochs ran:", epochs_ran)
print("MLP outputs:\n", outputs)
print("Predicted classes:\n", pred_classes)

# Plot SSE over epochs
plt.figure()
plt.plot(np.arange(1, len(sse_list)+1), sse_list)
plt.xlabel('Epoch')
plt.ylabel('Sum-squared-error')
plt.title('MLP training (AND) - Two output neurons')
plt.grid(True)
plt.show()
