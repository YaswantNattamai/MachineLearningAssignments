# SECTION A1: KNN Classification on Breast Cancer Data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("dataset.csv")
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
y = df["diagnosis"]
X = df.drop(columns=["diagnosis"])
X = X.fillna(X.mean(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
accuracy = neigh.score(X_test, y_test)
print("Accuracy (k=3):", accuracy)

print("\nTest Data Evaluation:")
y_test_pred = neigh.predict(X_test)
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))
print("Classification Report (Test):")
print(classification_report(y_test, y_test_pred))

test_acc = neigh.score(X_test, y_test)
train_acc = neigh.score(X_train, y_train)
if abs(train_acc - test_acc) > 0.1:
    if train_acc > test_acc:
        fit_type = "Overfitting"
    else:
        fit_type = "Underfitting"
else:
    fit_type = "Good Fit (Regular Fit)"
print(f"\nModel Fit Analysis: {fit_type}")


# SECTION A2: Linear Regression on Purchase Data

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

file_name = 'LabSessionData.xlsx'
df = pd.read_excel(file_name, sheet_name='Purchase data')
df = df[['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']]

np.random.seed(42)
num_synthetic = 900
synthetic_df = pd.DataFrame({
    'Candies (#)': np.random.randint(10, 30, size=num_synthetic),
    'Mangoes (Kg)': np.random.randint(1, 10, size=num_synthetic),
    'Milk Packets (#)': np.random.randint(1, 10, size=num_synthetic),
})
synthetic_df['Payment (Rs)'] = (
    synthetic_df['Candies (#)'] * 1 +
    synthetic_df['Mangoes (Kg)'] * 55 +
    synthetic_df['Milk Packets (#)'] * 18 +
    np.random.normal(0, 20, size=num_synthetic)
)
synthetic_df['Customer'] = [f"Fake_{i+1}" for i in range(num_synthetic)]
synthetic_df = synthetic_df[['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']]
combined_df = pd.concat([df, synthetic_df], ignore_index=True)

X = combined_df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
y = combined_df['Payment (Rs)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
print(f"R² Score: {r2:.4f}")
