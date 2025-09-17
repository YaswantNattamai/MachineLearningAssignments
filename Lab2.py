import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import statistics

def load_purchase_data(file_path="LabSessionData.xlsx"):
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

def interpret_matrix_rank(rank, dimension):
    """
    Interprets the meaning of the feature matrix rank in this context.
    """
    if rank == dimension:
        return (
            "The feature matrix has full rank. All products contribute "
            "independently to the purchase payment and their individual costs "
            "can be uniquely estimated."
        )
    else:
        return (
            f"The feature matrix rank ({rank}) is less than the dimension ({dimension}).\n"
            "There is redundancy or collinearity—at least one product's quantity is a linear "
            "combination of the others. The costs cannot be uniquely identified; "
            "infinite solutions exist."
        )

def get_dimension_and_vectors(X):
    """
    Returns the dimension (number of features) and number of data points.
    """
    return X.shape[1], X.shape[0]

# --------- A2: Rich/Poor Classifier ---------

def add_rich_poor_labels(df):
    """Add 'Class' column to DataFrame: RICH if payment > 200; else POOR."""
    df = df.copy()
    df['Class'] = np.where(df.iloc[:, 4] > 200, 'RICH', 'POOR')
    return df

def train_classifier(A, labels):
    """Train and evaluate a logistic regression classifier; returns classification report text."""
    X_train, X_test, y_train, y_test = train_test_split(A, labels, test_size=0.2, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report

# --------- A3: IRCTC Stock Data Analysis ---------

def load_irctc_stock_data(file_path="LabSessionData.xlsx"):
    """Load 'IRCTC Stock Price' worksheet as DataFrame."""
    return pd.read_excel(file_path, sheet_name='IRCTC Stock Price')

def get_price_mean_variance(df):
    """Return mean and variance of 'Price' column."""
    prices = df['Price']
    return statistics.mean(prices), statistics.variance(prices)

def wednesday_price_stats(df):
    """Return mean price for Wednesdays, and counts for Wednesday and all observations."""
    is_wed = df['Day'].str.startswith('Wed')
    wednesday_prices = df.loc[is_wed, 'Price']
    return statistics.mean(wednesday_prices), wednesday_prices.size, df.shape[0]

def april_price_mean(df):
    """Return mean price for April."""
    april_prices = df.loc[df['Month'] == 'Apr', 'Price']
    return statistics.mean(april_prices)

def loss_probability(df):
    """Return probability of loss (Chg% < 0) over the stock."""
    return (df['Chg%'] < 0).mean()

def profit_probability_wednesday(df):
    """Return probability of profit (Chg% > 0) on Wednesdays."""
    wed_mask = df['Day'].str.startswith('Wed')
    return (df.loc[wed_mask, 'Chg%'] > 0).mean()

def conditional_profit_given_wednesday(df):
    """Return P(Profit | Wednesday)."""
    wed_mask = df['Day'].str.startswith('Wed')
    return (df.loc[wed_mask, 'Chg%'] > 0).sum() / wed_mask.sum()

def plot_chg_vs_day(df):
    """Scatter plot: Chg% vs. Day of week."""
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df['Day'], y=df['Chg%'])
    plt.title('Chg% vs. Day of Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --------- A4: Thyroid Data Exploration ---------

def load_thyroid_data(file_path="LabSessionData.xlsx"):
    return pd.read_excel(file_path, sheet_name='thyroid0387_UCI', na_values=['?'])

def summarize_attributes(df):
    """Return summary table for each attribute: datatype, kind, encoding, missing, range."""
    summary = []
    for col in df.columns:
        col_data = df[col]
        dtype = col_data.dtype
        if dtype == object:
            uniq = set(col_data.dropna().unique())
            if uniq <= {'t', 'f'} or uniq <= {0, 1}:
                attr_type = 'binary'
                encoding = 'None'
            elif col_data.nunique() < 10:
                attr_type = 'nominal'
                encoding = 'One-hot'
            else:
                attr_type = 'categorical'
                encoding = 'Label'
            rng = None
        else:
            attr_type = 'numeric'
            encoding = 'None'
            rng = (col_data.min(), col_data.max())
        n_missing = col_data.isnull().sum() + np.sum(col_data == '?')
        summary.append({'column': col, 'dtype': str(dtype), 'attr_type': attr_type,
                        'encoding': encoding, 'missing': n_missing, 'range': rng})
    return pd.DataFrame(summary)

def numeric_stats(df):
    """Return mean and variance for numeric columns."""
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {'mean': df[col].mean(), 'variance': df[col].var()}
    return stats

def outlier_count(df):
    """Return count of outliers (outside 1.5*IQR) for each numeric column."""
    outliers = {}
    for col in df.select_dtypes(include=[np.number]):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        outliers[col] = mask.sum()
    return outliers

# --------- A5: Jaccard & SMC Similarity ---------

def get_first_two_binary_vectors(df):
    """Extract the first two binary observation vectors from binary columns."""
    binary_cols = [
        col for col in df.columns
        if set(df[col].dropna().unique()) <= {'t', 'f', 0, 1}
    ]
    binmat = df[binary_cols].replace({'t': 1, 'f': 0}).astype(int)
    return binmat.iloc[0].values, binmat.iloc[1].values

def jaccard_coefficient(vec1, vec2):
    """Return Jaccard coefficient for two binary vectors."""
    f11 = np.sum((vec1 == 1) & (vec2 == 1))
    f10 = np.sum((vec1 == 1) & (vec2 == 0))
    f01 = np.sum((vec1 == 0) & (vec2 == 1))
    denom = f11 + f10 + f01
    return f11 / denom if denom != 0 else np.nan

def smc_coefficient(vec1, vec2):
    """Return Simple Matching Coefficient for two binary vectors."""
    f11 = np.sum((vec1 == 1) & (vec2 == 1))
    f00 = np.sum((vec1 == 0) & (vec2 == 0))
    f10 = np.sum((vec1 == 1) & (vec2 == 0))
    f01 = np.sum((vec1 == 0) & (vec2 == 1))
    total = f11 + f00 + f10 + f01
    return (f11 + f00) / total if total != 0 else np.nan

# --------- A6: Cosine Similarity Function ---------

def cosine_similarity(A, B):
    """
    Compute the cosine similarity between two vectors A and B.
    """
    A = np.asarray(A).flatten()
    B = np.asarray(B).flatten()
    numerator = np.dot(A, B)
    denominator = np.linalg.norm(A) * np.linalg.norm(B)
    if denominator == 0:
        return 0.0
    return numerator / denominator

# --------- A7: Analyze Similarity Metrics ---------

def analyze_similarity_metrics(file_path, sheet_name, save_path=None):
    """
    Analyze and plot Cosine Similarity, Jaccard, and SMC matrices for first 20 observations.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, na_values=["?"])
    # Impute and encode categorical columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df[col] = df[col].fillna(df[col].mean())
    df_20 = df.iloc[:20].reset_index(drop=True)
    # Cosine Similarity
    cos_sim_matrix = np.zeros((20, 20))
    numeric_cols = df_20.select_dtypes(include=[np.number]).columns
    for i in range(20):
        for j in range(20):
            cos_sim_matrix[i, j] = cosine_similarity(df_20.loc[i, numeric_cols], df_20.loc[j, numeric_cols])
    # Identify binary columns
    binary_cols = [col for col in df_20.columns if set(df_20[col].unique()).issubset({0, 1})]
    def jaccard(a, b):
        f11 = np.sum((a == 1) & (b == 1))
        f10 = np.sum((a == 1) & (b == 0))
        f01 = np.sum((a == 0) & (b == 1))
        denom = f11 + f10 + f01
        return f11 / denom if denom != 0 else 0
    def smc(a, b):
        f11 = np.sum((a == 1) & (b == 1))
        f00 = np.sum((a == 0) & (b == 0))
        f10 = np.sum((a == 1) & (b == 0))
        f01 = np.sum((a == 0) & (b == 1))
        total = f11 + f00 + f10 + f01
        return (f11 + f00) / total if total != 0 else 0
    n = len(df_20)
    jc_matrix = np.zeros((n, n))
    smc_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vec1 = df_20.loc[i, binary_cols].values
            vec2 = df_20.loc[j, binary_cols].values
            jc_matrix[i, j] = jaccard(vec1, vec2)
            smc_matrix[i, j] = smc(vec1, vec2)
    # Plot heatmaps
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.heatmap(jc_matrix, annot=False, cmap='Blues', square=True, cbar=True)
    plt.title("Jaccard Coefficient")
    plt.subplot(1, 3, 2)
    sns.heatmap(smc_matrix, annot=False, cmap='Greens', square=True, cbar=True)
    plt.title("Simple Matching Coefficient")
    plt.subplot(1, 3, 3)
    sns.heatmap(cos_sim_matrix, annot=False, cmap='Reds', square=True, cbar=True)
    plt.title("Cosine Similarity")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# --------- A8: Imputation Summary ---------

def impute_missing_values(file_path, sheet_name, na_values=["?"], return_df=False):
    """Impute missing values with mode for categorical and mean/median for numeric columns."""
    df = pd.read_excel(file_path, sheet_name=sheet_name, na_values=na_values)
    filled_columns = []
    # Impute categorical columns with Mode
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(col.mode()[0]))
    filled_columns += [f"{col} (Mode)" for col in cat_cols if df[col].isnull().sum() == 0]
    def impute_numeric(col):
        if col.isnull().sum() == 0:
            return col
        q1, q3 = col.quantile([0.25, 0.75])
        iqr = q3 - q1
        has_outlier = ((col < (q1 - 1.5 * iqr)) | (col > (q3 + 1.5 * iqr))).any()
        method = "Median" if has_outlier else "Mean"
        filled_columns.append(f"{col.name} ({method})")
        return col.fillna(col.median() if has_outlier else col.mean())
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].apply(impute_numeric)
    print("Imputation complete.\nFilled columns and methods used:")
    for col in filled_columns:
        print(f"→ {col}")
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing == 0:
        print("\nAll missing values successfully imputed.")
    else:
        print(f"\nThere are still {remaining_missing} missing values remaining.")
    return df if return_df else None

# --------- A9: Min-Max Normalization ---------

def impute_and_normalize(file_path, sheet_name, na_values=["?"], return_df=False):
    """Impute missing values (mode/mean/median) and normalize numeric columns with Min-Max."""
    df = pd.read_excel(file_path, sheet_name=sheet_name, na_values=na_values)
    # Impute categorical columns with mode
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    # Impute numerical columns with mean or median based on outliers
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() == 0:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        has_outlier = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).any()
        df[col] = df[col].fillna(df[col].median() if has_outlier else df[col].mean())
    # Normalize numeric columns using Min-Max scaling
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Normalization complete. Sample normalized data:")
    print(df_normalized[numeric_cols].head())
    return df_normalized if return_df else None

# ==============================================
#                   MAIN OUTPUTS
# ==============================================
if __name__ == "__main__":
    # --- A1: Purchase Data Analysis
    A, C, purchase_df = load_purchase_data()
    dim, n_vecs, rk = get_vector_space_properties(A)
    costs = estimate_product_costs(A, C)
    print("A1: PURCHASE DATA ANALYSIS")
    print(f"Dimensionality (features): {dim}")
    print(f"Number of vectors (rows/customers): {n_vecs}")
    print(f"Rank of Purchase Quantity Matrix: {rk}")
    print(f"Estimated product costs: {costs}")

    coeffs = get_estimated_coefficients(costs)
    print("\nA6: Estimated per-unit product costs:")
    for k, v in coeffs.items():
        print(f"{k}: {v:.2f}")

    eqn = format_cost_equation(costs)
    print("\nA7: Estimated cost equation:")
    print(eqn)

    print("\nA8: Interpretation of matrix rank:")
    print(interpret_matrix_rank(rk, dim))

    dim2, nvec2 = get_dimension_and_vectors(A)
    print(f"\nA9: Dimension: {dim2}, Number of vectors: {nvec2}")

    # --- A2: Rich/Poor Classifier
    purchase_df = add_rich_poor_labels(purchase_df)
    report = train_classifier(A, purchase_df['Class'])
    print("\nA2: RICH/POOR CLASSIFICATION REPORT:")
    print(report)

    # --- A3: IRCTC Stock Data Analysis
    stock_df = load_irctc_stock_data()
    mean_price, var_price = get_price_mean_variance(stock_df)
    wed_mean, n_wed, n_all = wednesday_price_stats(stock_df)
    april_mean = april_price_mean(stock_df)
    loss_prob = loss_probability(stock_df)
    profit_wed_prob = profit_probability_wednesday(stock_df)
    cond_prob = conditional_profit_given_wednesday(stock_df)
    print("\nA3: IRCTC STOCK DATA ANALYSIS")
    print(f"Mean price: {mean_price:.2f}, Variance: {var_price:.2f}")
    print(f"Wednesday mean price: {wed_mean:.2f} (Wednesdays: {n_wed}, Total: {n_all})")
    print(f"April mean price: {april_mean:.2f}")
    print(f"Probability of Loss: {loss_prob:.2f}")
    print(f"Probability of Profit on Wednesday: {profit_wed_prob:.2f}")
    print(f"Conditional P(Profit|Wednesday): {cond_prob:.2f}")
    print("Plotting Chg% vs Day of Week...")
    plot_chg_vs_day(stock_df)

    # --- A4: Thyroid Data Summary
    thy_df = load_thyroid_data()
    summary_df = summarize_attributes(thy_df)
    num_stats = numeric_stats(thy_df)
    outliers = outlier_count(thy_df)
    print("\nA4: THYROID DATA SUMMARY")
    print(summary_df)
    print("Numeric mean and variance:\n", num_stats)
    print("Outlier count (numeric cols):\n", outliers)

    # --- A5: Jaccard & SMC Similarity on Binary Vectors
    vec1, vec2 = get_first_two_binary_vectors(thy_df)
    jc = jaccard_coefficient(vec1, vec2)
    smc = smc_coefficient(vec1, vec2)
    print("\nA5: SIMILARITY MEASURES for first two binary vectors")
    print(f"Jaccard coefficient: {jc:.3f}")
    print(f"Simple Matching coefficient: {smc:.3f}")
    if jc < smc:
        print("Jaccard is stricter: only positive matches; SMC includes negatives.")
    else:
        print("Both coefficients indicate similarity, but are used differently.")

    # --- A6: Cosine Similarity
    print("\nA6: Cosine similarity of first two numeric observations (thyroid dataset):")
    numeric_cols = thy_df.select_dtypes(include=[np.number]).columns
    vec1_num = thy_df.loc[0, numeric_cols].values
    vec2_num = thy_df.loc[1, numeric_cols].values
    cos_sim = cosine_similarity(vec1_num, vec2_num)
    print(f"Cosine Similarity: {cos_sim:.4f}")

    # --- A7: Analyze Similarity Metrics (can take time and display multiple plots)
    print("\nA7: Analyzing and plotting similarity metrics for first 20 thyroid records ...")
    analyze_similarity_metrics("LabSessionData.xlsx", "thyroid0387_UCI")

    # --- A8: Imputation Demonstration
    print("\nA8: Imputation demonstration (thyroid data):")
    impute_missing_values("LabSessionData.xlsx", "thyroid0387_UCI")

    # --- A9: Min-Max Normalization Demonstration
    print("\nA9: Min-Max normalization demonstration (thyroid data):")
    impute_and_normalize("LabSessionData.xlsx", "thyroid0387_UCI")
