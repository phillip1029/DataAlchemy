import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import gaussian_kde
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def generate_synthetic_data(df, n_samples, random_state=None, kde_bandwidth=None, correlation_threshold=0.1):
    """
    Generate synthetic data based on the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the original data.
        n_samples (int): Number of synthetic samples to generate.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.
        kde_bandwidth (float, optional): Bandwidth for KDE. Defaults to None (automatic selection).
        correlation_threshold (float, optional): Threshold for preserving correlations. Defaults to 0.1.

    Returns:
        pd.DataFrame: Synthetic data DataFrame.

    Raises:
        ValueError: If input DataFrame is empty or contains unsupported data types.
    """
    # Input validation
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    if not set(df.dtypes).issubset([np.int64, np.float64, object, 'category']):
        raise ValueError("Input DataFrame contains unsupported data types.")
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Separate numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Handle numerical features
    num_data = df[num_cols]
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    num_data_imputed = imputer.fit_transform(num_data)
    num_data_scaled = scaler.fit_transform(num_data_imputed)
    
    # Estimate the joint distribution of numerical features
    kde = gaussian_kde(num_data_scaled.T, bw_method=kde_bandwidth)
    
    # Generate synthetic numerical data
    synthetic_num_data = kde.resample(n_samples).T
    synthetic_num_data = scaler.inverse_transform(synthetic_num_data)
    synthetic_num_df = pd.DataFrame(synthetic_num_data, columns=num_cols)
    
    # Handle categorical features
    synthetic_cat_df = pd.DataFrame()
    for col in cat_cols:
        # Calculate probabilities of each category
        probs = df[col].value_counts(normalize=True)
        # Generate synthetic categorical data
        synthetic_cat_df[col] = np.random.choice(probs.index, size=n_samples, p=probs.values)
    
    # Combine numerical and categorical synthetic data
    synthetic_df = pd.concat([synthetic_num_df, synthetic_cat_df], axis=1)
    
    # Preserve relationships between numerical and categorical features
    for cat_col in cat_cols:
        encoder = TargetEncoder()
        for num_col in num_cols:
            encoded_col = encoder.fit_transform(df[cat_col], df[num_col]).squeeze()
            correlation = df[num_col].corr(encoded_col)
            if abs(correlation) > correlation_threshold:
                synthetic_df[num_col] += (encoder.transform(synthetic_df[cat_col]).squeeze() - encoded_col.mean()) * correlation
    
    return synthetic_df

def visualize_comparison(original_df, synthetic_df, num_cols=None, cat_cols=None):
    """
    Visualize the comparison between original and synthetic data.

    Args:
        original_df (pd.DataFrame): Original DataFrame.
        synthetic_df (pd.DataFrame): Synthetic DataFrame.
        num_cols (list, optional): List of numerical columns to visualize. Defaults to all numerical columns.
        cat_cols (list, optional): List of categorical columns to visualize. Defaults to all categorical columns.
    """
    if num_cols is None:
        num_cols = original_df.select_dtypes(include=['int64', 'float64']).columns
    
    if cat_cols is None:
        cat_cols = original_df.select_dtypes(include=['object', 'category']).columns
    
    # Visualize numerical columns
    for col in num_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(original_df[col], kde=True, label='Original', color='blue', alpha=0.5)
        sns.histplot(synthetic_df[col], kde=True, label='Synthetic', color='red', alpha=0.5)
        plt.title(f'Distribution Comparison: {col}')
        plt.legend()
        plt.show()
    
    # Visualize categorical columns
    for col in cat_cols:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=col, data=original_df, label='Original', alpha=0.5)
        sns.countplot(x=col, data=synthetic_df, label='Synthetic', alpha=0.5)
        plt.title(f'Distribution Comparison: {col}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master']
    })

    # Generate synthetic data
    synthetic_df = generate_synthetic_data(df, n_samples=100, random_state=42)

    print("Original Data:")
    print(df)
    print("\nSynthetic Data (first 5 rows):")
    print(synthetic_df.head())

    # Visualize comparison
    visualize_comparison(df, synthetic_df)