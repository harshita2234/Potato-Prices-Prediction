# Potato Prices Prediction

This repository contains the code and data for predicting potato prices based on various factors such as rainfall.

## Project Overview

This project aims to predict the prices of potatoes using historical data. The model can help farmers and traders make informed decisions by analysing patterns and trends.

## Features

- Data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Prediction and visualization

## Data

The data used in this project includes historical potato prices and rainfall data from various states. The final data used for training the model is stored in `final_potato_rainfall_data.csv`.

## Preprocessing Steps

Before training the model, the data is cleaned and preprocessed to ensure accuracy and reliability. The preprocessing steps are as follows:

```python
import pandas as pd

# Load the final output CSV file
final_data = pd.read_csv('path/to/file/initial_potato_rainfall_data.csv')

# Remove rows where any key field is NaN
final_data_cleaned = final_data.dropna(subset=['state', 'date', 'rainfall', 'price'])

# Save the cleaned final output back to a CSV file
final_data_cleaned.to_csv('path/to/file/final_potato_rainfall_data_cleaned.csv', index=False)

print("Data cleaning complete. Clean output saved to 'final_potato_rainfall_data_cleaned.csv'.")
```
## Installation

You need to have Python installed to run the code in this repository. You can install the necessary libraries using the following command:

```bash
pip install pandas==1.2.3 numpy==1.20.1 scikit-learn==0.24.1 matplotlib==3.3.4 seaborn==0.11.1
```
## Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/harshita2234/Potato-Prices-Prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Potato-Prices-Prediction
    ```
3. Ensure you have the cleaned data file in the appropriate directory:
    ```bash
    mv path/to/file/final_potato_rainfall_data_cleaned.csv .
    ```
4. Run the data preprocessing script:
    ```bash
    python preprocess_data.py
    ```
5. Train the model:
    ```bash
    python train_model.py
    ```
6. Make predictions:
    ```bash
    python predict.py
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

Thanks to all the contributors and data providers for their invaluable support in making this project possible.
