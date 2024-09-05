# SyntheticDataAlchemy

SyntheticDataAlchemy is a Python package for generating synthetic data that preserves the statistical properties and relationships of the original dataset.

## Installation

You can install SyntheticDataAlchemy using pip:

```
pip install SyntheticDataAlchemy
```

## Usage

Here's a simple example of how to use SyntheticDataAlchemy:

```python
import pandas as pd
from SyntheticDataAlchemy import generate_synthetic_data

# Load your original data
original_df = pd.read_csv('your_data.csv')

# Generate synthetic data
synthetic_df = generate_synthetic_data(original_df, n_samples=1000)

# Save the synthetic data
synthetic_df.to_csv('synthetic_data.csv', index=False)
```

## Features

- Generates synthetic data for both numerical and categorical features
- Preserves statistical properties of the original data
- Maintains relationships between features
- Handles missing data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.