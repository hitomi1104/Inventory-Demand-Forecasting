



# import streamlit as st
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# import xgboost as xgb
# from catboost import CatBoostRegressor
#
#
# # Function to load and prepare data based on aggregation type
# def load_and_prepare_data(df, aggregation_type='Monthly'):
#     df.drop(columns=['Sku'], inplace=True)  # Drop SKU column if not needed for prediction
#     df['Created'] = pd.to_datetime(df['Created'])
#
#     if aggregation_type == 'Monthly':
#         df['YearMonth'] = df['Created'].dt.to_period('M')  # Create a period for year-month
#         aggregated_data = df.groupby('YearMonth')['Quantity'].sum().reset_index()  # Sum the quantities for each month
#         aggregated_data['Month'] = aggregated_data['YearMonth'].dt.month  # Extract month
#         aggregated_data['Year'] = aggregated_data['YearMonth'].dt.year  # Extract year
#
#     elif aggregation_type == 'Weekly':
#         df['YearWeek'] = df['Created'].dt.to_period('W')  # Convert to a period with weekly frequency
#         aggregated_data = df.groupby('YearWeek')['Quantity'].sum().reset_index()  # Sum the quantities for each week
#         aggregated_data['Week'] = aggregated_data['YearWeek'].dt.week  # Extract week number
#         aggregated_data['Year'] = aggregated_data['YearWeek'].dt.year  # Extract year
#
#     elif aggregation_type == 'Daily':
#         df['Day'] = df['Created'].dt.date  # Extract date for daily aggregation
#         aggregated_data = df.groupby('Day')['Quantity'].sum().reset_index()  # Sum the quantities for each day
#         aggregated_data['Year'] = pd.to_datetime(aggregated_data['Day']).dt.year  # Extract year from date
#
#     elif aggregation_type == 'Quarterly':
#         df['Quarter'] = df['Created'].dt.to_period('Q')  # Create a period for year-quarter (e.g., '2024Q1')
#         aggregated_data = df.groupby('Quarter')['Quantity'].sum().reset_index()  # Sum the quantities for each quarter
#         aggregated_data['Year'] = aggregated_data['Quarter'].dt.year  # Extract year from quarter
#         aggregated_data['Quarter'] = aggregated_data['Quarter'].dt.quarter  # Extract the quarter number
#
#     elif aggregation_type == 'Yearly':
#         df['Year'] = df['Created'].dt.year  # Extract year for yearly aggregation
#         aggregated_data = df.groupby('Year')['Quantity'].sum().reset_index()  # Sum the quantities for each year
#
#     return aggregated_data
#
# ##################################################Plot##################################################
#
#
# # Function to plot the time series data using Seaborn
# def plot_timeseries(data, aggregation_type):
#     plt.figure(figsize=(12, 6))  # Increased figure size for readability
#
#     # Determine the correct column to plot
#     if aggregation_type == 'Daily':
#         data['Day'] = pd.to_datetime(data['Day'])
#         x_col = 'Day'
#         all_time_periods = pd.date_range(start=data['Day'].min(), end=data['Day'].max(), freq='D')
#         full_index_df = pd.DataFrame({x_col: all_time_periods})
#
#     elif aggregation_type == 'Weekly':
#         data['YearWeek'] = data['Year'].astype(str) + "-W" + data['Week'].astype(str).str.zfill(2)
#         data['YearWeek'] = pd.to_datetime(data['YearWeek'] + '-1', format='%Y-W%W-%w')
#         x_col = 'YearWeek'
#         all_time_periods = pd.date_range(start=data['YearWeek'].min(), end=data['YearWeek'].max(), freq='W-MON')
#         full_index_df = pd.DataFrame({x_col: all_time_periods})
#
#     elif aggregation_type == 'Monthly':
#         data['YearMonth'] = pd.to_datetime(data['YearMonth'].astype(str), format='%Y-%m').dt.to_period(
#             'M')  # Convert to Period
#         x_col = 'YearMonth'
#
#         # ✅ Use `period_range()` for full coverage & convert to string for plotting
#         all_time_periods = pd.period_range(start=data['YearMonth'].min(), end=data['YearMonth'].max(), freq='M').astype(
#             str)
#         full_index_df = pd.DataFrame({x_col: all_time_periods})
#
#         # ✅ Convert column to string before plotting
#         data[x_col] = data[x_col].astype(str)
#
#     elif aggregation_type == 'Quarterly':
#         data['Quarter'] = data['Year'].astype(str) + "Q" + data['Quarter'].astype(str)
#         x_col = 'Quarter'
#         all_time_periods = pd.period_range(start=data['Quarter'].min(), end=data['Quarter'].max(), freq='Q').astype(str)
#         full_index_df = pd.DataFrame({x_col: all_time_periods})
#
#     elif aggregation_type == 'Yearly':
#         data['Year'] = pd.to_datetime(data['Year'].astype(str) + '-01-01')
#         x_col = 'Year'
#         all_time_periods = pd.date_range(start=data['Year'].min(), end=data['Year'].max(), freq='Y')
#         full_index_df = pd.DataFrame({x_col: all_time_periods})
#
#     # ✅ Merge full time periods with original data and fill missing values
#     data = full_index_df.merge(data, on=x_col, how='left').fillna(0)
#
#     # ✅ Adjust bins dynamically for `displot()`
#     optimal_bins = min(50, max(5, len(data) // 5))  # Ensures bins scale with dataset size
#
#     # ✅ Plot based on aggregation type
#     if aggregation_type == 'Daily':
#         # 🔹 Daily: Use `lineplot()` for clear trends
#         sns.lineplot(data=data, x=x_col, y='Quantity', color='royalblue', linewidth=2)
#         plt.title(f'Line Plot of {aggregation_type} Quantity')
#
#     elif aggregation_type == 'Weekly':
#         # 🔹 Weekly: Use `barplot()` for clear visual separation
#         plt.figure(figsize=(12, 6))
#         sns.barplot(data=data, x=x_col, y="Quantity", color='royalblue')
#         plt.title(f'Weekly Aggregated Quantity')
#
#     else:
#         # 🔹 Other Aggregations: Use `displot()` with auto-adjusted bins
#         sns.displot(data, x=x_col, weights='Quantity', bins=optimal_bins, kde=False, color='royalblue', height=6, aspect=2)
#         plt.title(f'Distribution of {aggregation_type} Quantity')
#
#     plt.xlabel(aggregation_type)
#     plt.xticks(rotation=45)
#
#     # ✅ Format X-axis for readability
#     plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit number of labels
#
#     plt.ylabel('Quantity')
#     plt.grid(True)
#     st.pyplot(plt)  # Display the plot in Streamlit
#
#
#
# ################################################## Models ##################################################
#
# def cross_validate_models(data, aggregation_type, n_splits=5):
#     # Feature selection and preprocessing
#     if aggregation_type == 'Monthly':
#         X = data[['Month', 'Year']]
#         X['Month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
#         X['Month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)
#     elif aggregation_type == 'Weekly':
#         X = data[['Week', 'Year']]
#         X['Week_sin'] = np.sin(2 * np.pi * X['Week'] / 52)
#         X['Week_cos'] = np.cos(2 * np.pi * X['Week'] / 52)
#     elif aggregation_type == 'Daily':
#         X = data[['Year']]  # Remove 'Day' as it's datetime
#     elif aggregation_type == 'Quarterly':
#         X = data[['Quarter', 'Year']]
#         X['Quarter_sin'] = np.sin(2 * np.pi * X['Quarter'] / 4)
#         X['Quarter_cos'] = np.cos(2 * np.pi * X['Quarter'] / 4)
#     elif aggregation_type == 'Yearly':
#         X = data[['Year']]
#
#     y = data['Quantity']
#
#     # Rolling averages for trend detection
#     if len(X) > 6:
#         X['Rolling_Mean_3'] = y.rolling(3, min_periods=1).mean()
#         X['Rolling_Mean_6'] = y.rolling(6, min_periods=1).mean()
#
#     # Ensure X and y have the same number of rows
#     assert len(X) == len(y), f"Mismatch in number of samples: X({len(X)}) vs y({len(y)})"
#
#     # Standardize numerical features before train-test split
#     scaler = StandardScaler()
#     X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
#
#     # Define models
#     models = [
#         ('Linear Regression', LinearRegression()),
#         ('Ridge', Ridge()),
#         ('Lasso', Lasso()),
#         ('Elastic Net', ElasticNet()),
#         ('SVR', SVR()),
#         ('Decision Tree', DecisionTreeRegressor()),
#         ('Random Forest', RandomForestRegressor()),
#         ('Gradient Boosting', GradientBoostingRegressor()),
#         ('XGBoost', xgb.XGBRegressor(objective='reg:squarederror')),
#         ('CatBoost', CatBoostRegressor(verbose=0))
#     ]
#
#     results = {}
#
#     # Cross-validation setup
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#
#     # Cross-validation loop
#     for name, model in models:
#         mse_list, rmse_list, mae_list, r2_list = [], [], [], []
#
#         for train_index, test_index in kf.split(X_scaled):
#             X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
#             y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#             try:
#                 model.fit(X_train, y_train)
#                 predictions = model.predict(X_test)
#
#                 # Calculate metrics
#                 mse = mean_squared_error(y_test, predictions)
#                 rmse = np.sqrt(mse)
#                 mae = mean_absolute_error(y_test, predictions)
#                 r2 = r2_score(y_test, predictions)
#
#                 # Append metrics for each fold
#                 mse_list.append(mse)
#                 rmse_list.append(rmse)
#                 mae_list.append(mae)
#                 r2_list.append(r2)
#
#             except Exception as e:
#                 print(f"Model {name} failed to train in fold: {e}")
#                 continue
#
#         # Calculate RMSE standard deviation
#         rmse_std_dev = np.std(rmse_list)
#
#         # Store metrics for the model
#         results[name] = {
#             'MSE': round(mse, 1),  # Take average of MSE across all folds
#             'RMSE': round(rmse, 1),  # Average RMSE across all folds
#             'MAE': round(mae,1),  # Average MAE across all folds
#             'R²': round(r2,1),  # Average R² across all folds
#             'RMSE Std Dev': round(rmse_std_dev,1)  # RMSE standard deviation across all folds
#         }
#
#
#     # Convert results to DataFrame for better visualization
#     results_df = pd.DataFrame(results).T
#     results_df.index.name = 'Model'
#     results_df.reset_index(inplace=True)
#
#
#
#     return results_df
#
#
#
#
#
#
# ################################################## Output ##################################################
#
#     # Convert the results dictionary to a DataFrame for overall results
#     results_df = pd.DataFrame(results).T
#     results_df.index.name = 'Model'
#     results_df.reset_index(inplace=True)
#
#     # Convert the cross-validation fold results to a DataFrame
#     cross_val_df = pd.DataFrame(cross_val_results)
#
#     return results_df, cross_val_df
#
#
# # Streamlit UI to select a file and aggregation type
# st.image("images/uc3.png")
#
# # Path to the directory containing the CSV files
# file_directory = 'WFF/data/SKU2'
#
# # List all CSV files in the specified directory
# files = [f for f in os.listdir(file_directory) if f.endswith('.csv')]
#
# # Create a dropdown for file selection
# default_file = "0043121-X_2.csv"
# file_selected = st.selectbox("Select a SKU", files, index=files.index(default_file))
#
# # Check if a file is selected
# if file_selected:
#     st.write(f"You selected: {file_selected}")
#
#     # Load the data from the selected file
#     file_path = os.path.join(file_directory, file_selected)
#     df = pd.read_csv(file_path)
#
#     # Select aggregation type
#     aggregation_type = st.selectbox("Select aggregation type", ["Daily","Weekly", "Monthly", "Quarterly", "Yearly"])
#
#     # Load and prepare data based on the selected aggregation type
#     aggregated_data = load_and_prepare_data(df, aggregation_type)
#
#     # Show the aggregated data
#     st.write(f"Aggregated {aggregation_type} Data")
#     st.dataframe(aggregated_data, height=200)  # Adjust height as needed
#
#     # Plot the time series data
#     plot_timeseries(aggregated_data, aggregation_type)
#
#     # Perform cross-validation and get the model evaluation results
#     # results_df, cross_val_df = cross_validate_models(aggregated_data, aggregation_type)
#     results_df = cross_validate_models(aggregated_data, aggregation_type)
#
#     # Display the cross-validation results (summary table)
#     st.write("Model Evaluation Results (Average Metrics)", results_df)
#
#     # Display the detailed cross-validation results (RMSE per fold)
#     st.write("Cross-validation RMSE per fold", cross_val_df)
#
#     # Optionally, you can also allow the user to download the results as CSV
#     st.download_button(
#         label="Download Aggregated Data as CSV",
#         data=aggregated_data.to_csv(index=False),
#         file_name="aggregated_data.csv",
#         mime="text/csv"
#     )
#
#     st.download_button(
#         label="Download Model Evaluation Results as CSV",
#         data=results_df.to_csv(index=False),
#         file_name="model_evaluation_results.csv",
#         mime="text/csv"
#     )
#
#     st.download_button(
#         label="Download Cross-validation Results as CSV",
#         data=cross_val_df.to_csv(index=False),
#         file_name="cross_validation_results.csv",
#         mime="text/csv"
#     )



import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor


# Function to load and prepare data based on aggregation type
def load_and_prepare_data(df, aggregation_type='Monthly'):
    df.drop(columns=['Sku'], inplace=True)  # Drop SKU column if not needed for prediction
    df['Created'] = pd.to_datetime(df['Created'])

    if aggregation_type == 'Monthly':
        df['YearMonth'] = df['Created'].dt.to_period('M')  # Create a period for year-month
        aggregated_data = df.groupby('YearMonth')['Quantity'].sum().reset_index()  # Sum the quantities for each month
        aggregated_data['Month'] = aggregated_data['YearMonth'].dt.month  # Extract month
        aggregated_data['Year'] = aggregated_data['YearMonth'].dt.year  # Extract year

    elif aggregation_type == 'Weekly':
        df['YearWeek'] = df['Created'].dt.to_period('W')  # Convert to a period with weekly frequency
        aggregated_data = df.groupby('YearWeek')['Quantity'].sum().reset_index()  # Sum the quantities for each week
        aggregated_data['Week'] = aggregated_data['YearWeek'].dt.week  # Extract week number
        aggregated_data['Year'] = aggregated_data['YearWeek'].dt.year  # Extract year

    elif aggregation_type == 'Daily':
        df['Day'] = df['Created'].dt.date  # Extract date for daily aggregation
        aggregated_data = df.groupby('Day')['Quantity'].sum().reset_index()  # Sum the quantities for each day
        aggregated_data['Year'] = pd.to_datetime(aggregated_data['Day']).dt.year  # Extract year from date

    elif aggregation_type == 'Quarterly':
        df['Quarter'] = df['Created'].dt.to_period('Q')  # Create a period for year-quarter (e.g., '2024Q1')
        aggregated_data = df.groupby('Quarter')['Quantity'].sum().reset_index()  # Sum the quantities for each quarter
        aggregated_data['Year'] = aggregated_data['Quarter'].dt.year  # Extract year from quarter
        aggregated_data['Quarter'] = aggregated_data['Quarter'].dt.quarter  # Extract the quarter number

    elif aggregation_type == 'Yearly':
        df['Year'] = df['Created'].dt.year  # Extract year for yearly aggregation
        aggregated_data = df.groupby('Year')['Quantity'].sum().reset_index()  # Sum the quantities for each year

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
    elif aggregation_type == 'Daily':
        plt.plot(data['Day'].astype(str), data['Quantity'], marker='o', linestyle='-')
        plt.title('Time Series Plot of Quantity over Time (Daily)')
        plt.xlabel('Day')
    elif aggregation_type == 'Quarterly':
        plt.plot(data['Quarter'].astype(str), data['Quantity'], marker='o', linestyle='-')
        plt.title('Time Series Plot of Quantity over Time (Quarterly)')
        plt.xlabel('Quarter')
    elif aggregation_type == 'Yearly':
        plt.plot(data['Year'], data['Quantity'], marker='o', linestyle='-')
        plt.title('Time Series Plot of Quantity over Time (Yearly)')
        plt.xlabel('Year')

    plt.ylabel('Quantity')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)  # Display the plot in Streamlit


def cross_validate_models(data, aggregation_type):
    if aggregation_type == 'Monthly':
        X = data[['Month', 'Year']]  # Using 'Month' and 'Year' as features for monthly data
    elif aggregation_type == 'Weekly':
        X = data[['Week', 'Year']]  # Using 'Week' and 'Year' as features for weekly data
    elif aggregation_type == 'Daily':
        X = data[['Year', 'Day']]  # Using 'Year' and 'Day' as features for daily data
    elif aggregation_type == 'Quarterly':
        X = data[['Quarter', 'Year']]  # Using 'Quarter' and 'Year' as features for quarterly data
    elif aggregation_type == 'Yearly':
        X = data[['Year']]  # Using 'Year' as features for yearly data

    y = data['Quantity']

    # Ensure that X and y have the same number of rows
    assert len(X) == len(y), f"Mismatch in number of samples: X({len(X)}) vs y({len(y)})"

    # Check if we have enough samples for cross-validation
    if len(X) < 5:
        st.warning("Not enough samples for cross-validation. Skipping cross-validation.")
        return None, None
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

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
    cross_val_results = {}

    # Cross-validation loop to calculate the metrics
    for name, model in models:
        mse_list, rmse_list, mae_list, r2_list, rmse_std = [], [], [], [], []

        for fold_idx, (train_index, test_index) in enumerate(kf.split(X), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Fit the model
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Calculate metrics for each fold
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            # Append metrics for each fold
            mse_list.append(mse)
            rmse_list.append(rmse)
            mae_list.append(mae)
            r2_list.append(r2)

        rmse_std_dev = np.std(rmse_list)

        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'RMSE Std Dev': rmse_std_dev
        }

        # # Calculate average metrics across all folds
        # avg_mse = np.mean(mse_list)
        # avg_rmse = np.mean(rmse_list)
        # avg_mae = np.mean(mae_list)
        # avg_r2 = np.mean(r2_list)
        # rmse_std_dev = np.std(rmse_list)  # Standard deviation of RMSE across folds
        #
        # # Store the average metrics for the model in the main results table
        # results[name] = {
        #     'Average MSE': avg_mse,
        #     'Average RMSE': avg_rmse,
        #     'Average MAE': avg_mae,
        #     'Average R²': avg_r2,
        #     'RMSE Std Dev': rmse_std_dev
        # }

        # Store RMSE for cross-validation results table
        cross_val_results[name] = rmse_list

    # Convert the results dictionary to a DataFrame for overall results
    results_df = pd.DataFrame(results).T
    results_df.index.name = 'Model'
    results_df.reset_index(inplace=True)

    # Convert the cross-validation fold results to a DataFrame
    cross_val_df = pd.DataFrame(cross_val_results)

    return results_df, cross_val_df


# Streamlit UI to select a file and aggregation type
st.image("images/uc3.png")

# Path to the directory containing the CSV files
file_directory = 'WFF/data/SKU2'

# List all CSV files in the specified directory
files = [f for f in os.listdir(file_directory) if f.endswith('.csv')]

# Create a dropdown for file selection
default_file = "0043121-X_2.csv"
file_selected = st.selectbox("Select a SKU", files, index=files.index(default_file))

# Check if a file is selected
if file_selected:
    st.write(f"You selected: {file_selected}")

    # Load the data from the selected file
    file_path = os.path.join(file_directory, file_selected)
    df = pd.read_csv(file_path)

    # Select aggregation type
    aggregation_type = st.selectbox("Select aggregation type", ["Monthly", "Weekly", "Daily", "Quarterly", "Yearly"])

    # Load and prepare data based on the selected aggregation type
    aggregated_data = load_and_prepare_data(df, aggregation_type)

    # Show the aggregated data
    st.write(f"Aggregated {aggregation_type} Data", aggregated_data)

    # Plot the time series data
    plot_timeseries(aggregated_data, aggregation_type)

    # Perform cross-validation and get the model evaluation results
    results_df, cross_val_df = cross_validate_models(aggregated_data, aggregation_type)

    # Display the cross-validation results (summary table)
    st.write("Model Evaluation Results (Average Metrics)", results_df)

    # Display the detailed cross-validation results (RMSE per fold)
    st.write("Cross-validation RMSE per fold", cross_val_df)

    # Optionally, you can also allow the user to download the results as CSV
    st.download_button(
        label="Download Aggregated Data as CSV",
        data=aggregated_data.to_csv(index=False),
        file_name="aggregated_data.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Model Evaluation Results as CSV",
        data=results_df.to_csv(index=False),
        file_name="model_evaluation_results.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Cross-validation Results as CSV",
        data=cross_val_df.to_csv(index=False),
        file_name="cross_validation_results.csv",
        mime="text/csv"
    )











