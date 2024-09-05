import pandas as pd
from DataAlchemy import generate_synthetic_data

# Create a sample DataFrame
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master']
})

# Generate synthetic data
synthetic_df = generate_synthetic_data(df, n_samples=100)

print("Original Data:")
print(df)
print("\nSynthetic Data (first 5 rows):")
print(synthetic_df.head())

# Save the synthetic data
synthetic_df.to_csv('synthetic_data.csv', index=False)
print("\nSynthetic data saved to 'synthetic_data.csv'")