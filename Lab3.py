# A1: Euclidean Distance between Centroids
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("KNNAlgorithmDataset.csv")
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

features = df[["radius_mean", "texture_mean"]].values
labels = df["diagnosis"].values

features_M = features[labels == "M"]
features_B = features[labels == "B"]

centroid_M = np.mean(features_M, axis=0)
centroid_B = np.mean(features_B, axis=0)

spread_M = np.std(features_M, axis=0)
spread_B = np.std(features_B, axis=0)

interclass_distance = np.linalg.norm(centroid_M - centroid_B)
print(f"Euclidean distance between centroids: {interclass_distance:.2f}")

# A2: Histogram + Mean + Variance of First Feature
df = pd.read_csv("KNNAlgorithmDataset.csv")
feature_cols = df.columns[2:]
feature_cols = [col for col in feature_cols if not df[col].isnull().all()]
df[feature_cols] = df[feature_cols].apply(lambda col: col.fillna(col.mean()))

feature_name = feature_cols[0]
feature_data = df[feature_name].values.astype(float)
hist, bin_edges = np.histogram(feature_data, bins=10)

plt.figure(figsize=(8,5))
plt.hist(feature_data, bins=10, edgecolor='k', alpha=0.7)
plt.xlabel(feature_name)
plt.ylabel("Frequency")
plt.title(f"Histogram of {feature_name}")
plt.show()

mean = np.mean(feature_data)
variance = np.var(feature_data)

print("Feature:", feature_name)
print("Mean:", mean)
print("Variance:", variance)
