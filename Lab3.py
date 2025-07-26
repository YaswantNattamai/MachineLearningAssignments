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


# A3: Minkowski Distance Plot
df = pd.read_csv("KNNAlgorithmDataset.csv")
df = df.drop(columns=["id", "Unnamed: 32", "diagnosis"], errors="ignore")

x = df["radius_mean"].values
y = df["texture_mean"].values

r_values = [i for i in range(1,11)]
distances = [np.linalg.norm(x - y, ord=r) for r in r_values]

plt.plot(r_values, distances, marker='X', linestyle='-', color='red')
plt.title("Minkowski Distance between Two Vectors (r = 1 to 10)")
plt.xlabel("Order r")
plt.ylabel("Distance")
plt.xticks(r_values)
plt.savefig("LAB3question3plot.jpg")
plt.show()

# A4–A7: Training and Test KNN with k = 3
df = pd.read_csv("KNNAlgorithmDataset.csv")
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

y = df["diagnosis"]
X = df.drop(columns=["diagnosis"])
X = X.fillna(X.mean(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

accuracy = neigh.score(X_test, y_test)
print("Accuracy (k=3):", accuracy)

predictions = neigh.predict(X_test)
print("Predictions:", predictions)

# A8: Vary k from 1 to 11 and plot accuracy
k_values = range(1, 12)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracies.append(acc)
    print(f"k = {k} → Accuracy = {acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.title("KNN Accuracy for Different Values of k")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.savefig("KNN_accuracy_plot.jpg")
plt.show()

# A9: Confusion Matrix & Classification Report for Test and Train Data
print("\n--- A9: Classification Evaluation ---")

# Test Data Evaluation
print("\nTest Data Evaluation:")
y_test_pred = neigh.predict(X_test)
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))
print("Classification Report (Test):")
print(classification_report(y_test, y_test_pred))
