import numpy as np
import matplotlib.pyplot as plt
# A9: MLP for XOR gate (2-2-1, Sigmoid)
# -------------------------------
def mlp_train_XOR(max_epochs=5000, lr=0.1, tol=0.002):
    X = np.array([[0,0],[0,1],[1,0],[1,1]])   # XOR inputs
    T = np.array([[0],[1],[1],[0]])           # XOR targets
    N = X.shape[0]

    rng = np.random.default_rng(4)
    W1 = rng.normal(scale=0.5, size=(3,2))   # (bias+2 inputs → 2 hidden)
    W2 = rng.normal(scale=0.5, size=(3,1))   # (bias+2 hidden → 1 output)

    sse_list = []
    for epoch in range(max_epochs):
        # Forward pass
        Xb = np.hstack([np.ones((N,1)), X])
        net_h = Xb.dot(W1)
        out_h = 1/(1+np.exp(-net_h))   # sigmoid hidden
        out_hb = np.hstack([np.ones((N,1)), out_h])
        net_o = out_hb.dot(W2)
        out_o = 1/(1+np.exp(-net_o))   # sigmoid output

        err = T - out_o
        sse = 0.5*np.sum(err**2)
        sse_list.append(sse)

        if sse <= tol:
            break

        # Backpropagation
        delta_o = err * out_o * (1-out_o)
        grad_W2 = out_hb.T.dot(delta_o)

        delta_h = (delta_o.dot(W2[1:].T)) * out_h * (1-out_h)
        grad_W1 = Xb.T.dot(delta_h)

        # Weight updates
        W2 += lr * grad_W2
        W1 += lr * grad_W1

    return W1, W2, sse_list

# Train XOR MLP
W1_xor, W2_xor, sse_xor = mlp_train_XOR()
print("XOR MLP Final SSE:", sse_xor[-1], " Epochs:", len(sse_xor))

# Plot training curve
plt.figure(figsize=(8,5))
plt.plot(sse_xor)
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("A9: MLP Training on XOR (Sigmoid, 2-2-1 network)")
plt.grid(True)
plt.show()
