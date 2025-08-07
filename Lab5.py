
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv("Project_labeled_features.csv")

def prepare_regression_data(feature_columns, target_column="clarity_score", test_size=0.2, random_state=42):
    X = df[feature_columns]
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def task_A1_single_feature():
    X_train, X_test, y_train, y_test = prepare_regression_data(["mfcc_1","mfcc_2"])
    model = LinearRegression().fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

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

def task_A3_all_features():
    exclude_cols = ["id", "clarity_score", "clarity_label"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X_train, X_test, y_train, y_test = prepare_regression_data(feature_cols)
    model = LinearRegression().fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def task_A4_kmeans_k2():
    clustering_features = df.drop(columns=["id", "clarity_score", "clarity_label"])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(clustering_features)
    return kmeans, clustering_features


def task_A5_clustering_metrics(kmeans, X):
    sil = silhouette_score(X, kmeans.labels_)
    ch = calinski_harabasz_score(X, kmeans.labels_)
    db = davies_bouldin_score(X, kmeans.labels_)
    return {"Silhouette": sil, "Calinski-Harabasz": ch, "Davies-Bouldin": db}
