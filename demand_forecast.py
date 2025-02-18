import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import os


st.image("images/uc3.png", use_container_width=True)



# Path to the directory containing the CSV files
file_directory = 'WFF/data/SKU2'

# List all CSV files in the specified directory
files = [f for f in os.listdir(file_directory) if f.endswith('.csv')]

# Create a dropdown for file selection
default_file = "0043121-X_2.csv"
file_selected = st.selectbox("Select a file", files, index=files.index(default_file))
# file_selected = st.selectbox("Select a file", files)

# Check if a file is selected
if file_selected:
    st.write(f"You selected: {file_selected}")

    # Load the data from the selected file
    file_path = os.path.join(file_directory, file_selected)
    df = pd.read_csv(file_path)





# Streamlit dropdown for aggregation type (monthly or weekly)
aggregation_type = st.selectbox("Select Aggregation", ["Monthly", "Weekly"], index=0)


# Function to load and prepare data based on the aggregation type
def load_and_prepare_data(filepath, aggregation_type='Monthly'):
    df = pd.read_csv(filepath)
    df.drop(columns=['Sku'], inplace=True)  # Drop SKU column if not needed for prediction
    df['Created'] = pd.to_datetime(df['Created'])

    if aggregation_type == 'Monthly':
        # Aggregating by Month
        df['YearMonth'] = df['Created'].dt.to_period('M')  # Create a period for year-month
        aggregated_data = df.groupby('YearMonth')['Quantity'].sum().reset_index()  # Sum the quantities for each month
        aggregated_data['Month'] = aggregated_data['YearMonth'].dt.month  # Extract month
        aggregated_data['Year'] = aggregated_data['YearMonth'].dt.year  # Extract year
    elif aggregation_type == 'Weekly':
        # Aggregating by Week
        df['YearWeek'] = df['Created'].dt.to_period('W')  # Convert to a period with weekly frequency
        aggregated_data = df.groupby('YearWeek')['Quantity'].sum().reset_index()  # Sum the quantities for each week
        aggregated_data['Week'] = aggregated_data['YearWeek'].dt.week  # Extract week number
        aggregated_data['Year'] = aggregated_data['YearWeek'].dt.year  # Extract year

    return aggregated_data


# Function to plot the time series data
def plot_timeseries(data, aggregation_type):
    plt.figure(figsize=(10, 5))
    if aggregation_type == 'Monthly':
        plt.plot(data['YearMonth'].astype(str), data['Quantity'], marker='o', linestyle='-')
        plt.title('Time Series Plot of Quantity over Time (Monthly)')
        plt.xlabel('Month')
    elif aggregation_type == 'Weekly':
        plt.plot(data['YearWeek'].astype(str), data['Quantity'], marker='o', linestyle='-')
        plt.title('Time Series Plot of Quantity over Time (Weekly)')
        plt.xlabel('Week')

    plt.ylabel('Quantity')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)  # Use st.pyplot() to display the plot in Streamlit


# Function to build and evaluate models
def build_and_evaluate_models(data, aggregation_type):
    if aggregation_type == 'Monthly':
        X = data[['Month', 'Year']]  # Using 'Month' and 'Year' as features for monthly data
    elif aggregation_type == 'Weekly':
        X = data[['Week', 'Year']]  # Using 'Week' and 'Year' as features for weekly data

    y = data['Quantity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models list without GridSearch for simplicity
    models = [
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge()),
        ('Lasso', Lasso()),
        ('Elastic Net', ElasticNet()),
        ('SVR', SVR()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor()),
        ('Gradient Boosting', GradientBoostingRegressor()),
        ('XGBoost', xgb.XGBRegressor(objective='reg:squarederror')),
        ('CatBoost', CatBoostRegressor(verbose=0))
    ]

    results = {}

    for name, model in models:
        try:
            # Fit the model
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Metrics calculation
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
        except Exception as e:
            st.write(f"Error training {name}: {e}")

    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame(results).T
    results_df.index.name = 'Model'
    results_df.reset_index(inplace=True)

    return results_df


# Main Streamlit application logic
if file_selected:
    # Construct the file path
    file_path = os.path.join(file_directory, file_selected)

    # Load and prepare data based on the selected aggregation type
    data = load_and_prepare_data(file_path, aggregation_type)

    # Display the first few rows of data for the user
    st.write("### Data Preview")
    st.write(data.head())

    # Plot the time series
    st.write(f"### {aggregation_type} Time Series Plot")
    plot_timeseries(data, aggregation_type)

    # Build and evaluate models
    st.write("### Model Evaluation Results")
    results = build_and_evaluate_models(data, aggregation_type)
    st.write(results)