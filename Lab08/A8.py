import numpy as np
import matplotlib.pyplot as plt
# -------------------------------
# A8: MLP for AND gate
# -------------------------------
def mlp_train_AND(max_epochs=5000, lr=0.1, tol=0.002):
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    T = np.array([[0],[0],[0],[1]])
    N = X.shape[0]

    rng = np.random.default_rng(2)
    W1 = rng.normal(scale=0.5, size=(3,2))
    W2 = rng.normal(scale=0.5, size=(3,1))

    sse_list = []
    for epoch in range(max_epochs):
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

W1_and, W2_and, sse_and = mlp_train_AND()
print("AND MLP Final SSE:", sse_and[-1])

# Plot training curve
plt.figure(figsize=(8,5))
plt.plot(sse_and)
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("A8")
plt.grid(True)
plt.show()
