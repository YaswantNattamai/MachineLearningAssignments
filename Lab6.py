import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ================================ Question 1 & 2 ================================
def equal_width_binning(data, n_bins=4):
    """
    Perform equal-width binning on continuous numeric data.
    (question1_2.py)
    """
    data = np.array(data)
    min_val, max_val = data.min(), data.max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned_labels = np.digitize(data, bin_edges, right=False) - 1
    binned_labels[binned_labels == n_bins] = n_bins - 1
    return binned_labels, bin_edges

def calculate_entropy(data):
    """
    Calculate entropy of categorical or binned data.
    (question1_2.py)
    """
    values, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    return entropy

def calculate_gini(data):
    """
    Calculate Gini index of categorical or binned data.
    (question1_2.py)
    """
    values, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    gini = 1 - np.sum(probabilities ** 2)
    return gini

# ================================ Question 3 ================================
def equal_width_binning_q3(data, n_bins=4):
    """
    Converts continuous numeric data into categorical bins.
    (question3.py)
    """
    data = np.array(data)
    min_val, max_val = data.min(), data.max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned_labels = np.digitize(data, bin_edges, right=False) - 1
    binned_labels[binned_labels == n_bins] = n_bins - 1
    return binned_labels

def information_gain(df, feature, target):
    """
    Calculate information gain of a feature relative to the target.
    (question3.py)
    """
    total_entropy = calculate_entropy(df[target])
    values, counts = np.unique(df[feature], return_counts=True)
    weighted_entropy = 0
    for val, count in zip(values, counts):
        subset = df[df[feature] == val]
        weighted_entropy += (count / len(df)) * calculate_entropy(subset[target])
    return total_entropy - weighted_entropy

def find_root_node(df, target, n_bins=4):
    """
    Find root node based on information gain.
    (question3.py)
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if col != target and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = equal_width_binning_q3(df_copy[col], n_bins)
    if pd.api.types.is_numeric_dtype(df_copy[target]):
        df_copy[target] = equal_width_binning_q3(df_copy[target], n_bins)
    ig_scores = {}
    for col in df_copy.columns:
        if col != target:
            ig_scores[col] = information_gain(df_copy, col, target)
    root_feature = max(ig_scores, key=ig_scores.get)
    return root_feature, ig_scores

# ================================ Question 4 ================================
def binning(data, n_bins=4, method="equal_width"):
    """
    Converts continuous numeric data into categorical bins (flexible).
    (question4.py)
    """
    data = np.array(data)
    if method == "equal_width":
        min_val, max_val = data.min(), data.max()
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        binned_labels = np.digitize(data, bin_edges, right=False) - 1
        binned_labels[binned_labels == n_bins] = n_bins - 1
    elif method == "equal_frequency":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(data, quantiles)
        binned_labels = np.digitize(data, bin_edges, right=False) - 1
        binned_labels[binned_labels == n_bins] = n_bins - 1
    else:
        raise ValueError("Invalid method. Choose 'equal_width' or 'equal_frequency'.")
    return binned_labels, bin_edges

# ================================ Question 5 ================================
def calculate_entropy_q5(data):
    """
    Calculate entropy with lower numerical stability (question5.py).
    """
    values, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def information_gain_q5(feature, target):
    """
    Calculate information gain of a feature relative to the target (question5.py).
    """
    total_entropy = calculate_entropy_q5(target)
    values, counts = np.unique(feature, return_counts=True)
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset_target = target[feature == v]
        weighted_entropy += (c / len(feature)) * calculate_entropy_q5(subset_target)
    return total_entropy - weighted_entropy

class DecisionTree:
    """
    Simple decision tree implementation (question5.py).
    """
    def __init__(self, max_depth=3, n_bins=4, binning_method="equal_width"):
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.binning_method = binning_method
        self.tree = None

    def fit(self, X, y, depth=0):
        X = X.select_dtypes(include=[np.number])
        if calculate_entropy_q5(y) == 0 or depth == self.max_depth:
            unique, counts = np.unique(y, return_counts=True)
            return int(unique[np.argmax(counts)])
        best_feature = None
        best_gain = -1
        best_splits = None
        for feature in X.columns:
            binned_feature, _ = binning(X[feature], n_bins=self.n_bins, method=self.binning_method)
            gain = information_gain_q5(binned_feature, y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_splits = binned_feature
        if best_gain <= 0 or best_feature is None:
            unique, counts = np.unique(y, return_counts=True)
            return int(unique[np.argmax(counts)])
        tree = {best_feature: {}}
        for val in np.unique(best_splits):
            subset_X = X[best_splits == val]
            subset_y = y[best_splits == val]
            subtree = self.fit(subset_X, subset_y, depth + 1)
            tree[best_feature][int(val)] = subtree
        self.tree = tree
        return tree

    def predict_one(self, x, tree=None):
        if tree is None:
            tree = self.tree
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        feature_val = x[feature]
        binned_val, _ = binning([feature_val], n_bins=self.n_bins, method=self.binning_method)
        binned_val = int(binned_val[0])
        if binned_val in tree[feature]:
            return self.predict_one(x, tree[feature][binned_val])
        else:
            return None

    def predict(self, X):
        X = X.select_dtypes(include=[np.number])
        return [self.predict_one(row) for _, row in X.iterrows()]

def print_tree(tree, depth=0):
    """
    Pretty print decision tree (question5.py).
    """
    indent = " " * depth
    if not isinstance(tree, dict):
        print(f"{indent}→ {tree}")
        return
    for feature, branches in tree.items():
        for val, subtree in branches.items():
            print(f"{indent}[{feature} = {val}]")
            print_tree(subtree, depth + 1)

# ================================ Question 6 ================================
def plot_sklearn_decision_tree(df):
    """
    Train and plot sklearn decision tree classifier (cleaner visualization).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Separate features and target
    X = numeric_df.drop(columns=["mfcc_1", "clarity_score"], errors="ignore")
    y = numeric_df["mfcc_1"]
    
    # Bin target variable
    y_binned = pd.cut(y, bins=4, labels=[0,1,2,3])
    
    # Train decision tree
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
    clf.fit(X, y_binned)
    
    # Pick top 5 features by importance
    importances = clf.feature_importances_
    important_features = X.columns[np.argsort(importances)[::-1][:5]]
    
    # Retrain tree with only top 5 features
    X_top = X[important_features]
    clf.fit(X_top, y_binned)
    
    # Plot
    plt.figure(figsize=(30,15))
    plot_tree(
        clf,
        feature_names=X_top.columns,
        class_names=[str(c) for c in np.unique(y_binned)],
        filled=True,
        rounded=True,
        fontsize=12
    )
    plt.show()

# ================================ Question 7 ================================
def plot_decision_boundary(df):
    """
    Plot decision tree decision boundary using two chosen features (question7.py).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    X = numeric_df[['dct_4', 'mfcc_2']]
    y = numeric_df['mfcc_1']
    y_binned = pd.cut(y, bins=4, labels=[0, 1, 2, 3])
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
    clf.fit(X, y_binned)
    x_min, x_max = X['dct_4'].min() - 1, X['dct_4'].max() + 1
    y_min, y_max = X['mfcc_2'].min() - 1, X['mfcc_2'].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X['dct_4'], X['mfcc_2'],
                c=y_binned.astype(int), edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.xlabel('dct_4')
    plt.ylabel('mfcc_2')
    plt.title('Decision Tree Decision Boundary (dct_4 vs mfcc_2)')
    plt.show()

# ===== LOAD DATASET =====
df = pd.read_csv("Project_labeled_features.csv")

# ========== Question 1 ==========
print("Q1: Equal-width binning and entropy/gini calculation")
binned_scores, bin_edges = equal_width_binning(df['mfcc_1'], n_bins=4)
entropy_val = calculate_entropy(binned_scores)
gini_index = calculate_gini(binned_scores)
print("Bin edges:", bin_edges)
print("First 10 binned scores:", binned_scores[:10])
print("Entropy:", entropy_val)
print("Gini_index:", gini_index)
print("\n")


# ========== Question 3 ==========
print("Q3: Find root node with highest information gain (binned features)")
root_feature, ig_scores = find_root_node(df, target="mfcc_1", n_bins=4)
print("Root Node Feature:", root_feature)
print("Information Gain Scores:")
for feat, score in ig_scores.items():
    print(f"{feat}: {score:.4f}")
print("\n")

# ========== Question 4 ==========
print("Q4: Flexible binning (equal-width and equal-frequency)")
binned_width, edges_width = binning(df['mfcc_1'], n_bins=4, method="equal_width")
binned_freq, edges_freq = binning(df['mfcc_1'], n_bins=4, method="equal_frequency")
print("Equal Width Binning -> First 10:", binned_width[:10])
print("Bin Edges:", edges_width)
print("Equal Frequency Binning -> First 10:", binned_freq[:10])
print("Bin Edges:", edges_freq)
print("\n")

# ========== Question 5 ==========
print("Q5: Custom Decision Tree implementation")
X_q5 = df.drop(columns=['mfcc_1']).select_dtypes(include=[np.number])
y_q5, _ = binning(df['mfcc_1'], n_bins=4, method="equal_width")
dt = DecisionTree(max_depth=3, n_bins=4, binning_method="equal_width")
tree = dt.fit(X_q5, y_q5)
print("Decision Tree Structure:")
print(tree)
print("Pretty Print Decision Tree:")
print_tree(tree)
print("\n")


# ========== Question 6 ==========
print("Q6: Sklearn Decision Tree visualization")
plot_sklearn_decision_tree(df)  # Shows matplotlib plot

# ========== Question 7 ==========
print("Q7: Plot Decision Boundary for two selected features")
plot_decision_boundary(df)  # Shows matplotlib plot of decision boundary
