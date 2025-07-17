
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import statistics


def load_purchase_data(file_path="Lab-Session-Data.xlsx"):
    """Load the 'Purchase data' worksheet and return feature and label matrices."""
    df = pd.read_excel(file_path, sheet_name='Purchase data')
    X = df.iloc[:, 1:4].values  # Quantities of Candies, Mangoes, Milk Packets
    y = df.iloc[:, 4].values.reshape(-1, 1)  # Payment column
    return X, y, df

def get_vector_space_properties(A):
    """Return (dimension, n_vectors, rank) of the matrix A."""
    dimension = A.shape[1]
    n_vectors = A.shape[0]
    rank = np.linalg.matrix_rank(A)
    return dimension, n_vectors, rank

def estimate_product_costs(A, C):
    """Estimate product costs using Moore-Penrose pseudo-inverse."""
    costs = np.linalg.pinv(A) @ C
    return costs.flatten()

def get_estimated_coefficients(product_costs):
    """Return the estimated coefficients for the cost equation."""
    return {
        'Candies (per piece)': product_costs[0],
        'Mangoes (per kg)': product_costs[1],
        'Milk Packets (per packet)': product_costs[2]
    }

def format_cost_equation(product_costs):
    """Formats the estimated cost equation as a string."""
    return (
        f"Estimated Payment = "
        f"{product_costs[0]:.2f} * Candies + "
        f"{product_costs[1]:.2f} * Mangoes + "
        f"{product_costs[2]:.2f} * Milk Packets"
    )
