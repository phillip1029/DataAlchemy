import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import gaussian_kde
from category_encoders import TargetEncoder

def generate_synthetic_data(df, n_samples):
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
    kde = gaussian_kde(num_data_scaled.T)
    
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
            encoded_col = encoder.fit_transform(df[cat_col], df[num_col])
            correlation = df[num_col].corr(encoded_col)
            if abs(correlation) > 0.1:  # Adjust this threshold as needed
                synthetic_df[num_col] += (encoder.transform(synthetic_df[cat_col]) - encoded_col.mean()) * correlation
    
    return synthetic_df