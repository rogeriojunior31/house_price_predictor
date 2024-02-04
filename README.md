## Overview

This project aims to predict housing prices in California using the California Housing Dataset. It employs a linear regression model, enhanced with cross-validation and hyperparameter tuning for improved accuracy. Additionally, the project includes data visualization to analyze the relationship between house features and prices and to compare actual prices with model predictions.

## Requirements

- Python 3.8 or higher
- Scikit-learn (for machine learning models)
- NumPy (for numerical computations)
- Matplotlib (for plotting graphs)
- Seaborn (for enhanced data visualization)

To install the required packages, run:

bashCopy code

`pip install numpy matplotlib seaborn scikit-learn`

## Project Structure

The project is structured into two main classes:

1. `CaliforniaHousePricePredictor`: A base class that loads the dataset, trains a simple linear regression model, and evaluates its performance using RMSE (Root Mean Square Error).
    
2. `EnhancedCaliforniaHousePricePredictor`: A derived class that extends the base class by incorporating cross-validation and hyperparameter tuning using Ridge regression and `GridSearchCV`. It also includes methods for visualizing data and model predictions.
    

### Key Features

- **Data Loading and Splitting**: Automatically loads the California Housing Dataset and splits it into training and testing sets.
    
- **Model Training**: Trains a linear regression model, with an enhanced version employing Ridge regression for regularization and using cross-validation for model validation.
    
- **Model Evaluation**: Evaluates the model's performance on the test set and calculates the RMSE to quantify prediction accuracy.
    
- **Data Visualization**: Includes functions to visualize the relationship between house features (e.g., median income of residents) and house prices, and to compare actual prices with predicted prices.
    

## Usage

To use this project, follow these steps:

1. **Clone or download this repository** to your local machine.
    
2. **Navigate to the project directory** and install the required packages (if not already installed).
    
3. **Run the project** by executing the Python script containing the classes and methods described above. The main execution will instantiate the `EnhancedCaliforniaHousePricePredictor` class and run the model training, evaluation, and visualization processes.
    

## Example Output

- RMSE value indicating the model's prediction accuracy.
- Scatter plots showing the relationship between a selected house feature and its price.
- Scatter plot comparing actual vs. predicted house prices.

## Contributing

Contributions to this project are welcome. You can contribute by improving the model's accuracy, adding new visualization features, or suggesting other enhancements.

## License

This project is open-source and available under the MIT License.