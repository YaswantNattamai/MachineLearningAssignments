import numpy as np
# A11: sklearn MLPClassifier
# -------------------------------
from sklearn.neural_network import MLPClassifier

X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
T_and = np.array([0,0,0,1])
T_step = np.array([0,0,0,1]) 
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])   # XOR inputs
T_xor = np.array([0, 1, 1, 0])           # XOR targets
# AND
clf_and = MLPClassifier(hidden_layer_sizes=(2,), activation="logistic", solver="sgd",
                        learning_rate_init=0.1, max_iter=5000, random_state=1)
clf_and.fit(X_and, T_step)
print("sklearn MLP (AND) predictions:", clf_and.predict(X_and))

# XOR
clf_xor = MLPClassifier(hidden_layer_sizes=(2,), activation="logistic", solver="lbfgs",
                        learning_rate_init=0.1, max_iter=1000, random_state=1)
clf_xor.fit(X_xor, T_xor)
print("sklearn MLP (XOR) predictions:", clf_xor.predict(X_xor))
