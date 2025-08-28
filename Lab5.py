
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv("Project_labeled_features.csv")
'''
A1. If your project deals with a regression problem, please use one attribute of your dataset 
(X_train) along with the target values (y_train) for training a linear regression model. Sample code 
suggested below.

'''
def prepare_regression_data(feature_columns, target_column="clarity_score", test_size=0.2, random_state=42):
    X = df[feature_columns]
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def task_A1_single_feature():
    X_train, X_test, y_train, y_test = prepare_regression_data(["mfcc_1","mfcc_2"])
    model = LinearRegression().fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


'''

A2. Calculate MSE, RMSE, MAPE and R2 scores for prediction made by the trained model in A1.  
Perform prediction on the test data and compare the metric values between train and test set

'''



def task_A2_evaluate_single(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics = {}
    for name, (true, pred) in {
        "train": (y_train, y_train_pred),
        "test": (y_test, y_test_pred)
    }.items():
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(true, pred)
        r2 = r2_score(true, pred)
        metrics[name] = {"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}
    return metrics

'''
A3. Repeat the exercises A1 and A2 with more than one attribute or all attributes. 
'''
def task_A3_all_features():
    exclude_cols = ["id", "clarity_score", "clarity_label"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X_train, X_test, y_train, y_test = prepare_regression_data(feature_cols)
    model = LinearRegression().fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

'''
A4. Perform k-means clustering on your data. Please remove / ignore the target variable for 
performing clustering. Sample code suggested below. 
'''



def task_A4_kmeans_k2():
    clustering_features = df.drop(columns=["id", "clarity_score", "clarity_label"])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(clustering_features)
    return kmeans, clustering_features

'''
A5. For the clustering done in A4, calculate the: (i) Silhouette Score, (ii) CH Score and (iii) DB Index. 

'''
def task_A5_clustering_metrics(kmeans, X):
    sil = silhouette_score(X, kmeans.labels_)
    ch = calinski_harabasz_score(X, kmeans.labels_)
    db = davies_bouldin_score(X, kmeans.labels_)
    return {"Silhouette": sil, "Calinski-Harabasz": ch, "Davies-Bouldin": db}

'''
A6. Perform k-means clustering for different values of k. Evaluate the above scores for each k value. 
Make a plot of the values against the k value to determine the optimal cluster count. 
'''
def task_A6_kmeans_diff_k(k_values=range(2, 8)):
    clustering_features = df.drop(columns=["id", "clarity_score", "clarity_label"])
    results = {}
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(clustering_features)
        sil = silhouette_score(clustering_features, kmeans.labels_)
        ch = calinski_harabasz_score(clustering_features, kmeans.labels_)
        db = davies_bouldin_score(clustering_features, kmeans.labels_)
        results[k] = {"Silhouette": sil, "Calinski-Harabasz": ch, "Davies-Bouldin": db}
    return results


'''
A7. Using elbow plot, determine the optimal k value for k-means clustering.
'''
def task_A7_elbow_plot(max_k=10):
    clustering_features = df.drop(columns=["id", "clarity_score", "clarity_label"])
    distortions = []
    k_values = range(2, max_k+1)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(clustering_features)
        distortions.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, distortions, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (Distortion)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    model_single, X_train_s, X_test_s, y_train_s, y_test_s = task_A1_single_feature()
    metrics_single = task_A2_evaluate_single(model_single, X_train_s, X_test_s, y_train_s, y_test_s)
    print("A2 Metrics (Single Feature):")
    for set_type, scores in metrics_single.items():
        print(f"  {set_type.capitalize()} Set:")
        for metric, value in scores.items():
            print(f"    {metric}: {value:.4f}")
    model_all, X_train_a, X_test_a, y_train_a, y_test_a = task_A3_all_features()
    metrics_all = task_A2_evaluate_single(model_all, X_train_a, X_test_a, y_train_a, y_test_a)
    print("\nA3 Metrics (All Features):")
    for set_type, scores in metrics_all.items():
        print(f"  {set_type.capitalize()} Set:")
        for metric, value in scores.items():
            print(f"    {metric}: {value:.4f}")
    kmeans2, X_cluster = task_A4_kmeans_k2()
    cluster_metrics = task_A5_clustering_metrics(kmeans2, X_cluster)
    print("\nA5 Clustering Metrics:")
    for metric, value in cluster_metrics.items():
        print(f"  {metric}: {value:.4f}")
    k_results = task_A6_kmeans_diff_k(range(2, 6))
    print("\nA6 Metrics for Different k:")
    for k, scores in k_results.items():
        print(f"  k={k}:")
        for metric, value in scores.items():
            print(f"    {metric}: {value:.4f}")
    print("\nA7 Elbow Plot:")
    task_A7_elbow_plot(max_k=10)
    print("Elbow plot generated. Check the plot for optimal k.")