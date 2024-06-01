# Potato Prices Prediction

This repository contains the code and data for predicting potato prices based on various factors such as rainfall.

## Project Overview

This project aims to predict the prices of potatoes using historical data. The model can help farmers and traders make informed decisions by analysing patterns and trends.

## Project Steps

1. **Data Collection and Preprocessing**: We used three datasets - `potato.csv`, `rainfall_news.csv`, and `state.csv` to create the final dataset `final_potato_rainfall_data.csv`. `preprocessing-steps.py` explain how to do so.
2. **Data Cleaning**: The final dataset was cleaned to ensure accuracy and reliability. The steps are as follows:

    ```python
    import pandas as pd

    # Load the final output CSV file
    final_data = pd.read_csv('final_potato_rainfall_data.csv')

    # Remove rows where any key field is NaN
    final_data_cleaned = final_data.dropna(subset=['state', 'date', 'rainfall', 'price'])

    # Save the cleaned final output back to a CSV file
    final_data_cleaned.to_csv('final_potato_rainfall_data_cleaned.csv', index=False)

    print("Data cleaning complete. Clean output saved to 'final_potato_rainfall_data_cleaned.csv'.")
    ```
3. **Modeling**: The cleaned data was used to train the following models:
    - K-Nearest Neighbors (KNN)
    - Long Short-Term Memory (LSTM)
    - Random Forest Regressor

## Installation

You need to have Python installed to run the code in this repository. You can install the necessary libraries using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
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
    mv final_potato_rainfall_data_cleaned.csv .
    ```
4. Run the models:
    ```bash
    python knn.py
    python lstm.py
    python regressor.py
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

Thanks to all the contributors and data providers for their invaluable support in making this project possible.
