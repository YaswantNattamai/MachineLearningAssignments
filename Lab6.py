import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


df = pd.read_csv("Project_labeled_features.csv")

target_col = "clarity_label"

# A1. Entropy calculation
def entropy(series):
    """Calculate entropy of a categorical pandas Series."""
    probs = series.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))

# Equal-width binning
def equal_width_binning(series, bins=4):
    """Convert continuous data into categorical using equal-width binning."""
    return pd.cut(series, bins=bins, labels=False)

# Example: binning Clarity_Score if needed
df["Clarity_Score_binned"] = equal_width_binning(df["Clarity_Score"], bins=4)


# A2. Gini Index
def gini_index(series):
    """Calculate Gini index for a categorical pandas Series."""
    probs = series.value_counts(normalize=True)
    return 1 - np.sum(probs**2)

# A3. Feature with max Information Gain
def information_gain(df, feature, target):
    """Calculate information gain for a given feature."""
    total_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0
    for val in values:
        subset = df[df[feature] == val]
        weighted_entropy += (len(subset)/len(df)) * entropy(subset[target])
    return total_entropy - weighted_entropy
