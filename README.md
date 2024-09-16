# DataAlchemy

## Introduction

DataAlchemy is a powerful tool for generating synthetic data that closely mimics your original dataset. This README provides an overview of DataAlchemy's features and usage from a business user's perspective.

## Why Use DataAlchemy?

1. **Data Privacy**: Generate synthetic data that preserves the statistical properties of your original data without exposing sensitive information.
2. **Augment Limited Datasets**: Expand small datasets for more robust analysis and machine learning model training.
3. **Test Data Generation**: Create realistic test data for software development and quality assurance processes.
4. **Scenario Analysis**: Generate data for various "what-if" scenarios to support decision-making.

## Features

### 1. Synthetic Data Generation

DataAlchemy can create synthetic versions of your datasets that maintain the statistical properties and relationships of the original data.

### 2. Configurability

- **Sample Size Control**: Generate any number of synthetic samples.
- **Reproducibility**: Set a random seed for consistent results across multiple runs.
- **Fine-tuning**: Adjust parameters like KDE bandwidth and correlation threshold for more precise control over the synthetic data generation process.

### 3. Visualization

Compare your original data with the generated synthetic data using built-in visualization tools. This helps ensure that the synthetic data accurately represents the characteristics of your original dataset.

## Installation

1. Ensure you have Python installed on your system.
2. Clone this repository or download the DataAlchemy package.
3. Open a terminal or command prompt in the DataAlchemy directory.
4. Run the following command to install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## How to Use DataAlchemy

### Step 1: Prepare Your Data

Organize your data into a pandas DataFrame. DataAlchemy supports both numerical and categorical data types.

### Step 2: Generate Synthetic Data

Here's a basic example of how to use DataAlchemy to generate synthetic data:

```python
from SyntheticDataAlchemy.synthetic_data import generate_synthetic_data, visualize_comparison
import pandas as pd

# Load your original data
original_data = pd.read_csv('your_data.csv')

# Generate synthetic data
synthetic_data = generate_synthetic_data(original_data, n_samples=1000)

# Compare original and synthetic data
visualize_comparison(original_data, synthetic_data)
```

### Step 3: Customize the Process (Optional)

You can fine-tune the synthetic data generation process using additional parameters:

```python
synthetic_data = generate_synthetic_data(
    original_data,
    n_samples=1000,
    random_state=42,  # Set for reproducibility
    kde_bandwidth=0.1,  # Adjust for smoother or more detailed distributions
    correlation_threshold=0.2  # Adjust for preserving weaker or stronger relationships
)
```

### Step 4: Analyze and Validate

Use the `visualize_comparison` function to compare the distributions of your original and synthetic data. This will help you ensure that the synthetic data accurately represents your original dataset.

## Best Practices

1. **Start Small**: Begin with a subset of your data to quickly test and understand the synthetic data generation process.
2. **Iterate**: Adjust parameters and regenerate data as needed to achieve the desired balance between privacy and utility.
3. **Validate Thoroughly**: Always compare the synthetic data with your original data to ensure it meets your requirements.
4. **Document Your Process**: Keep track of the parameters used to generate your synthetic datasets for reproducibility and audit purposes.

## Conclusion

DataAlchemy provides a user-friendly way to generate high-quality synthetic data for various business needs. By following this guide, you can leverage DataAlchemy to create valuable synthetic datasets while preserving the privacy and integrity of your original data.

For more detailed information on the functions and parameters, please refer to the inline documentation in the `synthetic_data.py` file.
