import numpy as np
import matplotlib.pyplot as plt


def step_activation(net):
    return 1.0 if net >= 0 else 0.0

def bipolar_step_activation(net):
    return 1.0 if net >= 0 else -1.0

def sigmoid_activation(net):
    return 1/(1+np.exp(-net))

def tanh_activation(net):
    return np.tanh(net)

def relu_activation(net):
    return np.maximum(0, net)

# Error
def sum_squared_error(y_true, y_pred):
    return 0.5*np.sum((np.array(y_true)-np.array(y_pred))**2)


# Perceptron Training (Step / Bipolar)
def perceptron_train(X, T, w_init, lr=0.1, activation="step", max_epochs=100):
    X = np.asarray(X)
    N, n = X.shape
    Xb = np.hstack([np.ones((N,1)), X])  # add bias column
    w = np.array(w_init, dtype=float).copy()
    sse_list = []

    if activation == "step":
        act_fn = step_activation
    elif activation == "bipolar":
        act_fn = bipolar_step_activation
    else:
        raise ValueError("Use perceptron_train only for step/bipolar")

    for epoch in range(max_epochs):
        outputs = []
        for i in range(N):
            net = np.dot(w, Xb[i])
            y = act_fn(net)
            outputs.append(y)
            delta = (T[i] - y)
            w = w + lr * delta * Xb[i]
        sse = sum_squared_error(T, outputs)
        sse_list.append(sse)
        if sse <= 0.002:
            break
    return w, sse_list


# Gradient Descent Training (Sigmoid, Tanh, ReLU)
def grad_descent_train(X, T, w_init, lr=0.1, activation="sigmoid", max_epochs=200):
    X = np.asarray(X)
    N = X.shape[0]
    Xb = np.hstack([np.ones((N,1)), X])
    w = np.array(w_init, dtype=float).copy()
    sse_list = []

    for epoch in range(max_epochs):
        net = Xb.dot(w)

        if activation == "sigmoid":
            out = sigmoid_activation(net)
            deriv = out*(1-out)
        elif activation == "tanh":
            out = tanh_activation(net)
            deriv = 1 - out**2
        elif activation == "relu":
            out = relu_activation(net)
            deriv = (net > 0).astype(float)
        else:
            raise ValueError("Unsupported activation in grad_descent_train")

        errors = T - out
        sse = 0.5*np.sum(errors**2)
        sse_list.append(sse)

        if sse <= 0.002:
            break

        delta = errors * deriv
        grad_w = Xb.T.dot(delta)
        w = w + lr * grad_w
    return w, sse_list


# Run Experiments (A3)
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])

# Targets
T_step = np.array([0,0,0,1])        # Step perceptron targets
T_bipolar = np.array([-1,-1,-1,1])  # Bipolar perceptron targets
T_cont = np.array([0,0,0,1])        # Sigmoid/Tanh/ReLU targets

# Initial weights
w_init = np.zeros(3)

results = {}

# Step
_, sse_step = perceptron_train(X_and, T_step, [0.1,0.1,0.1], lr=0.2, activation="step")
results["Step"] = sse_step

# Bipolar Step
_, sse_bipolar = perceptron_train(X_and, T_bipolar, [0.1,0.1,0.1], lr=0.2, activation="bipolar")
results["Bipolar Step"] = sse_bipolar

# Sigmoid
_, sse_sigmoid = grad_descent_train(X_and, T_cont, w_init, lr=0.5, activation="sigmoid", max_epochs=200)
results["Sigmoid"] = sse_sigmoid

# Tanh
_, sse_tanh = grad_descent_train(X_and, T_cont, w_init, lr=0.5, activation="tanh", max_epochs=200)
results["Tanh"] = sse_tanh

# ReLU
_, sse_relu = grad_descent_train(X_and, T_cont, w_init, lr=0.1, activation="relu", max_epochs=200)
results["ReLU"] = sse_relu

# Plot Results
plt.figure(figsize=(10,6))
for name, sse in results.items():
    plt.plot(sse, label=name)
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("A3: AND Gate Training with Different Activations")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# A4: Learning rate vs Epochs (AND, Step activation)
# -------------------------------
lrs = [0.1 * i for i in range(1, 11)]
epochs_record = []

for lr in lrs:
    _, sse_list = perceptron_train(X_and, T_step, [0.1,0.1,0.1], lr=lr, activation="step", max_epochs=1000)
    epochs_record.append(len(sse_list))

plt.figure(figsize=(8,5))
plt.plot(lrs, epochs_record, marker="o")
plt.xlabel("Learning Rate")
plt.ylabel("Epochs to Converge")
plt.title("A4: Learning Rate vs Epochs (AND Gate, Step activation)")
plt.grid(True)
plt.show()

print("Learning rate vs Epochs:", dict(zip(lrs, epochs_record)))

# -------------------------------
# A5: XOR Gate
# -------------------------------
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
T_xor = np.array([0,1,1,0])   # standard XOR targets

# (1) Single-layer perceptron attempts
_, sse_step_xor = perceptron_train(X_xor, T_xor, [0.1,0.1,0.1], lr=0.2, activation="step", max_epochs=50)
_, sse_bipolar_xor = perceptron_train(X_xor, np.array([-1,1,1,-1]), [0.1,0.1,0.1], lr=0.2, activation="bipolar", max_epochs=50)

print("XOR with Step activation SSE (last epoch):", sse_step_xor[-1])
print("XOR with Bipolar Step SSE (last epoch):", sse_bipolar_xor[-1])
print("Observation: They cannot converge since XOR is not linearly separable.")

# (2) Multi-layer Perceptron (2-2-1) with Sigmoid
def mlp_train_XOR(max_epochs=5000, lr=0.5, tol=0.002):
    X = X_xor
    T = T_xor.reshape(-1,1)
    N = X.shape[0]

    rng = np.random.default_rng(1)
    W1 = rng.normal(scale=0.5, size=(3,2))  # (bias+2 inputs -> 2 hidden)
    W2 = rng.normal(scale=0.5, size=(3,1))  # (bias+2 hidden -> 1 output)

    sse_list = []
    for epoch in range(max_epochs):
        # Forward pass
        Xb = np.hstack([np.ones((N,1)), X])
        net_h = Xb.dot(W1)
        out_h = 1/(1+np.exp(-net_h))
        out_hb = np.hstack([np.ones((N,1)), out_h])
        net_o = out_hb.dot(W2)
        out_o = 1/(1+np.exp(-net_o))

        err = T - out_o
        sse = 0.5*np.sum(err**2)
        sse_list.append(sse)

        if sse <= tol:
            break

        # Backprop
        delta_o = err * out_o * (1-out_o)
        grad_W2 = out_hb.T.dot(delta_o)

        delta_h = (delta_o.dot(W2[1:].T)) * out_h * (1-out_h)
        grad_W1 = Xb.T.dot(delta_h)

        W2 += lr * grad_W2
        W1 += lr * grad_W1

    return W1, W2, sse_list

W1, W2, sse_xor_mlp = mlp_train_XOR()
print("Final SSE (MLP for XOR):", sse_xor_mlp[-1], " Epochs:", len(sse_xor_mlp))

plt.figure(figsize=(8,5))
plt.plot(sse_xor_mlp)
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("A5: MLP Training on XOR (Sigmoid, 2-2-1 network)")
plt.grid(True)
plt.show()


# -------------------------------
# A6: Customer Transaction Dataset
# -------------------------------
cust_X = np.array([
    [20,6,2,386], [16,3,6,289], [27,6,2,393], [19,1,2,110], [24,4,2,280],
    [22,1,5,167], [15,4,2,271], [18,4,2,274], [21,1,4,148], [16,2,4,198]
], dtype=float)

cust_T = np.array([1,1,1,0,1,0,1,1,0,0], dtype=float)

# Min-max normalization
minv, maxv = cust_X.min(axis=0), cust_X.max(axis=0)
cust_Xs = (cust_X - minv) / (maxv - minv + 1e-9)

# Train sigmoid perceptron
w_init_cust = np.zeros(cust_Xs.shape[1]+1)
w_cust, sse_cust = grad_descent_train(cust_Xs, cust_T, w_init_cust, lr=0.1, activation="sigmoid", max_epochs=5000)

# Predictions
Xb_c = np.hstack([np.ones((cust_Xs.shape[0],1)), cust_Xs])
probs = 1/(1+np.exp(-Xb_c.dot(w_cust)))
preds = (probs >= 0.5).astype(int)

print("True labels: ", cust_T.astype(int))
print("Pred labels: ", preds)

# -------------------------------
# A7: Pseudo-inverse vs Gradient descent
# -------------------------------
from numpy.linalg import pinv

# Pseudo-inverse solution
Xb_c = np.hstack([np.ones((cust_Xs.shape[0],1)), cust_Xs])
w_pinv = pinv(Xb_c).dot(cust_T)

# Predictions
probs_pinv = 1/(1+np.exp(-Xb_c.dot(w_pinv)))
preds_pinv = (probs_pinv >= 0.5).astype(int)

print("Pseudo-inverse preds:", preds_pinv)
print("Perceptron preds    :", preds)

