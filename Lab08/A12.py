# A12: Project dataset with sklearn MLP
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load CSV
df = pd.read_csv("student_features_with_labels.csv")

# Drop non-feature columns
X = df.drop(columns=["File Name", "Source_File", "Number", "Clarity"])
y = df["Clarity"]

# Encode target labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'Low'->0, 'High'->1, etc.

# Optional: Scale features (recommended for MLP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Initialize MLP classifier
clf_proj = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42
)

# Train
clf_proj.fit(X_train, y_train)

# Training accuracy
train_acc = clf_proj.score(X_train, y_train)
print("Training accuracy:", train_acc)

# Test accuracy
y_pred = clf_proj.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("Test accuracy:", test_acc)
