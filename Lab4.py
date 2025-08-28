# SECTION A1: KNN Classification on Breast Cancer Data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


'''
A1. Confusion matrix for your classification problem. 
From confusion matrix, the other performance metrics such as precision, recall and F1-Score measures for both training and test 
data. infer the models learning outcome (underfit / regularfit / overfit). 
'''
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

'''
A2. Calculate MSE, RMSE, MAPE and R2 scores for the price prediction exercise done in Lab 02. Analyse the results. 
'''
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


'''
A3. Generate 20 data points (training set data) consisting of 2 features (X & Y) whose values vary 
randomly between 1 & 10. Based on the values, assign these 20 points to 2 different classes (class0 - 
Blue & class1 - Red). Make a scatter plot of the training data and color the points as per their class 
color. Observe the plot. 

'''
# SECTION A3: Plot 20 Random Labeled Points in 2D

import matplotlib.pyplot as plt

np.random.seed(42)
X_vals = np.random.uniform(1, 10, 20)
Y_vals = np.random.uniform(1, 10, 20)
labels = np.random.choice([0, 1], size=20)
df_plot = pd.DataFrame({'X': X_vals, 'Y': Y_vals, 'Label': labels})

plt.figure(figsize=(8, 6))
for label, color in zip([0, 1], ['blue', 'red']):
    subset = df_plot[df_plot['Label'] == label]
    plt.scatter(subset['X'], subset['Y'], c=color, label=f'Class {label}', s=60)
plt.title('A3: 20 Training Points with 2 Classes')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()


'''
A4. Generate test set data with values of X & Y varying between 0 and 10 with increments of 0.1. 
This creates a test set of about 10,000 points. Classify these points with above training data using 
kNN classifier (k = 3). Make a scatter plot of the test data output with test points colored as per their 
predicted class colors (all points predicted class0 are labeled blue color). Observe the color spread 
and class boundary lines in the feature space.

'''
# SECTION A4: kNN Classification (k=3) of 2D Grid

from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
X_train_vals = np.random.uniform(1, 10, 20)
Y_train_vals = np.random.uniform(1, 10, 20)
train_labels = np.random.choice([0, 1], size=20)
train_df = pd.DataFrame({'X': X_train_vals, 'Y': Y_train_vals, 'Label': train_labels})

x_range = np.arange(0, 10.1, 0.1)
y_range = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_range, y_range)
X_test_points = np.c_[xx.ravel(), yy.ravel()]

knn = KNeighborsClassifier(n_neighbors=3)
X_train = train_df[['X', 'Y']].values
y_train = train_df['Label'].values
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test_points)

plt.figure(figsize=(8, 6))
plt.scatter(X_test_points[:, 0], X_test_points[:, 1],
            c=['blue' if label == 0 else 'red' for label in y_pred],
            alpha=0.3, s=10, marker='s', label='Predicted Region')
for label, color in zip([0, 1], ['blue', 'red']):
    subset = train_df[train_df['Label'] == label]
    plt.scatter(subset['X'], subset['Y'], c=color, edgecolors='black',
                label=f'Train Class {label}', s=80)
plt.title("A4: kNN Classification (k=3) of Test Grid")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

'''
A5. Repeat A4 exercise for various values of k and observe the change in the class boundary lines. 

'''

# SECTION A5: kNN Classification with Varying k on 2D Grid



np.random.seed(42)
X_train_vals = np.random.uniform(1, 10, 20)
Y_train_vals = np.random.uniform(1, 10, 20)
train_labels = np.random.choice([0, 1], size=20)
train_df = pd.DataFrame({'X': X_train_vals, 'Y': Y_train_vals, 'Label': train_labels})

x_range = np.arange(0, 10.1, 0.1)
y_range = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_range, y_range)
X_test_points = np.c_[xx.ravel(), yy.ravel()]

k_values = [1, 2, 3, 4, 5, 7, 11]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    X_train = train_df[['X', 'Y']].values
    y_train = train_df['Label'].values
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test_points)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_points[:, 0], X_test_points[:, 1],
                c=['blue' if label == 0 else 'red' for label in y_pred],
                alpha=0.3, s=10, marker='s')
    for label, color in zip([0, 1], ['blue', 'red']):
        subset = train_df[train_df['Label'] == label]
        plt.scatter(subset['X'], subset['Y'], c=color, edgecolors='black',
                    label=f'Train Class {label}', s=80)
    plt.title(f"A5: Decision Boundary with k = {k}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

'''
A6. Repeat the exercises A3 to A5 for your project data considering any two features and classes. 

'''


# SECTION A6: kNN Decision Boundaries on Real Dataset Subset


import matplotlib.gridspec as gridspec

data = pd.read_csv("dataset.csv")
features = ['radius_mean', 'texture_mean']
X = data[features].values
y = (data['diagnosis'] == 'M').astype(int).values
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=3, stratify=y)

np.random.seed(7)
subset_idx = np.random.choice(X_train.shape[0], min(40, X_train.shape[0]), replace=False)
X_sub, y_sub = X_train[subset_idx], y_train[subset_idx]
x_min, x_max = X_sub[:, 0].min() - 1, X_sub[:, 0].max() + 1
y_min, y_max = X_sub[:, 1].min() - 1, X_sub[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

ks = [1, 3, 5, 7, 9]
plt.figure(figsize=(4*len(ks), 4))
gs = gridspec.GridSpec(1, len(ks))
for i, k in enumerate(ks):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_sub, y_sub)
    Z = knn.predict(grid_points).reshape(xx.shape)
    ax = plt.subplot(gs[i])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    ax.scatter(X_sub[:, 0], X_sub[:, 1], c=y_sub, cmap='bwr', edgecolor='k', s=70)
    ax.set_title(f"kNN Boundary k={k}")
    ax.set_xlabel(features[0])
    if i == 0:
        ax.set_ylabel(features[1])
    ax.grid(True)
plt.suptitle("kNN Boundaries on Real Dataset Subset")
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

'''
A7. Use RandomSearchCV() or GridSearchCV() operations to find the ideal ‘k’ value for your 
kNN classifier. This is called hyper-parameter tuning. 
'''

# A7: kNN Classification with Cross-Validation for Optimal k
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("dataset.csv")
features = ['radius_mean', 'texture_mean']
X = data[features].values
y = (data['diagnosis'] == 'M').astype(int).values
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.25, random_state=3, stratify=y
)
param_grid = {'n_neighbors': np.arange(1, 21)}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Optimal k: {grid.best_params_['n_neighbors']} with cross-validation score: {grid.best_score_:.3f}")

