# modular_viva_ml.py

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Function: Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=["id", "clarity_score", "clarity_label"])
    y = df["clarity_label"]

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Function: Hyperparameter tuning
def tune_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    param_dist_rf = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    rf_random = RandomizedSearchCV(
        rf, param_distributions=param_dist_rf, n_iter=10, cv=5,
        scoring="accuracy", random_state=42, n_jobs=-1
    )
    rf_random.fit(X_train, y_train)
    print("Random Forest Best Parameters:", rf_random.best_params_)
    return rf_random.best_params_


# Function: Train and evaluate models

def evaluate_models(X_train, X_test, y_train, y_test, rf_best_params):
    models = {
        "SVM": SVC(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(**rf_best_params),
        "AdaBoost": AdaBoostClassifier(),
        "XGBoost": XGBClassifier(eval_metric="mlogloss"),
        "CatBoost": CatBoostClassifier(verbose=0),
        "NaiveBayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=500)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, y_train_pred),
            "Test Accuracy": accuracy_score(y_test, y_test_pred),
            "Precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_test_pred, average="weighted"),
            "F1-Score": f1_score(y_test, y_test_pred, average="weighted"),
        })

    results_df = pd.DataFrame(results)
    return results_df


# Main workflow

if __name__ == "__main__":
    file_path = "labeled_features.csv"
    X_train, X_test, y_train, y_test = load_data(file_path)
    rf_best_params = tune_random_forest(X_train, y_train)
    results_df = evaluate_models(X_train, X_test, y_train, y_test, rf_best_params)
    print(results_df)