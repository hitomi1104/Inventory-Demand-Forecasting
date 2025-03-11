import streamlit as st

import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.regression.linear_model import OLS
from scipy.stats import skew
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import datetime
from pathlib import Path
import random

# Scikit-Learn models:

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import catboost as cbt
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor




# LSTM:

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
# from keras.utils import np_utils
from keras.utils import to_categorical
from keras.layers import LSTM


# ARIMA Model:

import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse


import pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def sales_aggregation(data, time="monthly"):
    data = data.copy()

    # Convert 'Date' column to datetime format
    data["Date"] = pd.to_datetime(data["Date"])

    # Aggregate based on selected time period
    if time == "monthly":
        data["Date"] = data["Date"].dt.to_period("M").astype(str)
    elif time == "yearly":
        data["Date"] = data["Date"].dt.to_period("Y").astype(str)
    elif time == "weekly":
        data["Date"] = data["Date"].dt.to_period("W").apply(lambda x: x.start_time.strftime('%Y-%m-%d'))
    elif time == "daily":
        data["Date"] = data["Date"].dt.date  # Keeps full date
    else:
        raise ValueError("Invalid time period. Choose from 'daily', 'weekly', 'monthly', or 'yearly'.")

    # Sum shipped quantity per selected time period
    data = data.groupby("Date")["ShippedQty"].sum().reset_index()

    # Convert date column back to datetime format (for correct plotting)
    if time in ["monthly", "yearly"]:
        data["Date"] = pd.to_datetime(data["Date"])

    return data


def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    overfitting_results = []
    predictions = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2})

        # Overfitting check (Compare Train vs. Test)
        y_train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        overfitting_results.append({
            "Model": name,
            "Train MAE": train_mae, "Test MAE": mae,
            "Train RMSE": train_rmse, "Test RMSE": rmse,
            "Train R²": train_r2, "Test R²": r2
        })

    # Convert results into DataFrames
    df_results = pd.DataFrame(results)
    df_overfitting_results = pd.DataFrame(overfitting_results)

    return df_results, df_overfitting_results, predictions

# Function to train, predict, and visualize using Plotly
def plot_each_model_predictions(y_train, y_test, model, dates_train, dates_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Create interactive Plotly figure
    fig = go.Figure()

    # Add actual train & test data
    fig.add_trace(go.Scatter(x=dates_train, y=y_train, mode='lines', name="Train Data (Original)", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=dates_test, y=y_test, mode='lines', name="Test Data (Original)", line=dict(color="black")))

    # Add model predictions
    fig.add_trace(go.Scatter(x=dates_test, y=y_pred, mode='lines', name=f"{model_name} Prediction", line=dict(color="red", dash="dot")))

    # Compute error metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Update layout
    fig.update_layout(
        title=f"{model_name} Forecasting Prediction",
        xaxis_title="Date",
        yaxis_title="Sales",
        legend=dict(x=0, y=1),
        annotations=[
            dict(
                x=dates_test.iloc[0],
                y=max(y_test),
                text=f"RMSE: {rmse:.2f}<br>MAE: {mae:.2f}<br>R²: {r2:.4f}",
                showarrow=False,
                font=dict(size=12),
                bgcolor="white"
            )
        ]
    )

    st.plotly_chart(fig)


# Function to generate future forecasts
def future_forecast(model):
    future_steps = 7  # Define forecast period
    dates_future = pd.date_range(start=dates_test.iloc[-1], periods=future_steps + 1, freq='D')[1:]

    # Train model on full training data
    model.fit(X_train, y_train)

    # Prepare future input features
    future_X = np.zeros((future_steps, X_train.shape[1]))
    future_X[0, :] = X_test[-1, :]  # Start with last test sample

    future_predictions = []
    for i in range(future_steps):
        next_pred = model.predict(future_X[i, :].reshape(1, -1))[0]
        future_predictions.append(next_pred)

        # Shift features forward
        if i + 1 < future_steps:
            future_X[i + 1, :-1] = future_X[i, 1:]
            future_X[i + 1, -1] = next_pred

    # Convert to NumPy array
    y_pred_future = np.array(future_predictions)

    # Aggregate predictions to monthly forecast
    weekly_forecast = y_pred_future.sum()
    # monthly_forecast = y_pred_future.sum()  # Sum daily predictions for a monthly forecast

    # Create future forecast visualization
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=dates_future, y=y_pred_future, mode='lines', name="Future Forecast",
                                    line=dict(color="green", dash="dot")))
    fig_future.update_layout(title="Future Sales Forecast", xaxis_title="Date", yaxis_title="Sales")

    return fig_future, int(weekly_forecast)






######################################################################################################################################

######################################################################################################################################

######################################################################################################################################




st.set_page_config(page_title="Sales Dashboard", layout="centered")
st.image("images/uc3.png")


file_directory = 'data/SKUs'
files = [f for f in os.listdir(file_directory) if f.endswith('.csv')]

# Create a dropdown for file selection
default_file = "50.csv"
file_selected = st.selectbox("Select a SKU", files, index=files.index(default_file))
if file_selected:
    st.write(f"You selected: {file_selected}")
    file_path = os.path.join(file_directory, file_selected)
    df = pd.read_csv(file_path)
    

m_df = sales_aggregation(df, "monthly")
y_df = sales_aggregation(df, "yearly")
w_df = sales_aggregation(df, "weekly")
d_df = sales_aggregation(df, "daily")

st.write(d_df,  height=300)

# Streamlit UI
st.subheader("Distributions of ShippedQty")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
# Yearly Sales
with col1:
    fig, ax = plt.subplots(figsize=(4, 3))
    y_df.plot(kind="bar", x="Date", y="ShippedQty", color="mediumblue", ax=ax)
    ax.set_title("Yearly Sales")
    ax.set_xlabel("Years")
    ax.set_ylabel("Shipped Qty")
    ax.set_xticklabels(y_df["Date"].dt.year, rotation=0)
    st.pyplot(fig)

# Monthly Sales
with col2:
    fig, ax = plt.subplots(figsize=(4, 3))
    m_df.plot(kind="bar", x="Date", y="ShippedQty", color="mediumblue", ax=ax)
    ax.set_title("Monthly Sales")
    ax.set_xlabel("Months")
    ax.set_ylabel("Shipped Qty")
    tick_interval = max(1, len(m_df) // 10)
    ax.set_xticks(np.arange(0, len(m_df), tick_interval))
    ax.set_xticklabels(m_df["Date"].dt.strftime('%Y-%m')[::tick_interval], rotation=45)
    st.pyplot(fig)

# Weekly Sales
with col3:
    fig, ax = plt.subplots(figsize=(4, 3))
    w_df.plot(marker="o", x="Date", y="ShippedQty", color="darkorange", ax=ax)
    ax.set_title("Weekly Sales")
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Shipped Qty")
    ax.set_xticks(range(0, len(w_df), max(1, len(w_df) // 10)))
    ax.set_xticklabels(w_df["Date"][::max(1, len(w_df) // 10)], rotation=45)
    st.pyplot(fig)

# Daily Sales
with col4:
    fig, ax = plt.subplots(figsize=(4, 3))
    d_df.plot(x="Date", y="ShippedQty", color="darkorange", ax=ax)
    ax.set_title("Daily Sales")
    ax.set_xlabel("Days")
    ax.set_ylabel("Shipped Qty")
    ax.set_xticklabels(d_df["Date"][::max(1, len(d_df) // 10)], rotation=45)
    st.pyplot(fig)

# Formatting
sns.despine()


######################################################################################################################################
# skewness = d_df["ShippedQty"].skew()
#
# # Streamlit UI
# st.subheader("Distribution of ShippedQty")
# st.write(f"**Skewness of ShippedQty:** {skewness:.2f}")
#
# # Create figure for visualization
# fig, axes = plt.subplots(1, 2, figsize=(8, 3))
#
# # Histogram with KDE
# sns.histplot(d_df["ShippedQty"], bins=30, kde=True, ax=axes[0])
# axes[0].set_title("PDF")
#
# # Boxplot
# sns.boxplot(x=d_df["ShippedQty"], ax=axes[1])
# axes[1].set_title("Boxplot")
#
# # Display in Streamlit
# st.pyplot(fig)

######################################################################################################################################

# Scaling summary
X = d_df[["ShippedQty"]].values

# Apply different scalers
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

d_df["MinMaxScaled"] = minmax_scaler.fit_transform(X)
d_df["StandardScaled"] = standard_scaler.fit_transform(X)
d_df["RobustScaled"] = robust_scaler.fit_transform(X)


######################################################################################################################################
# Feature Engineeing
# Prepare d_df for Machine Learning Models 
d_df_ml = d_df.copy()

# Extract basic date-based features
d_df_ml["Date"] = pd.to_datetime(d_df_ml["Date"])
d_df_ml["DayOfWeek"] = d_df_ml["Date"].dt.dayofweek
d_df_ml["Month"] = d_df_ml["Date"].dt.month

# Add lag features using ShippedQty_Sqrt
d_df_ml["Lag_1"] = d_df_ml["ShippedQty"].shift(1)
d_df_ml["Lag_7"] = d_df_ml["ShippedQty"].shift(7)
d_df_ml["Lag_14"] = d_df_ml["ShippedQty"].shift(14)
d_df_ml["Lag_30"] = d_df_ml["ShippedQty"].shift(30)

# Add rolling statistics using ShippedQty_Sqrt
d_df_ml["Rolling_7"] = d_df_ml["ShippedQty"].rolling(window=7).mean()
d_df_ml["Rolling_7_Std"] = d_df_ml["ShippedQty"].rolling(window=7).std()

# Add cyclical encoding for date-based features
d_df_ml["DayOfWeek_Sin"] = np.sin(2 * np.pi * d_df_ml["DayOfWeek"] / 7)
d_df_ml["DayOfWeek_Cos"] = np.cos(2 * np.pi * d_df_ml["DayOfWeek"] / 7)
d_df_ml["Month_Sin"] = np.sin(2 * np.pi * d_df_ml["Month"] / 12)
d_df_ml["Month_Cos"] = np.cos(2 * np.pi * d_df_ml["Month"] / 12)

# Add feature interactions using ShippedQty_Sqrt
d_df_ml["Lag_1_to_Lag_7_Ratio"] = d_df_ml["Lag_1"] / (d_df_ml["Lag_7"] + 1e-5)
d_df_ml["Lag_7_to_Lag_14_Ratio"] = d_df_ml["Lag_7"] / (d_df_ml["Lag_14"] + 1e-5)

# Drop NaN values after feature engineering
d_df_ml = d_df_ml.dropna().reset_index(drop=True)



# Prepare d_df for Neural Networks
d_df_nn = d_df.copy()

# Convert Date column to datetime
d_df_nn["Date"] = pd.to_datetime(d_df_nn["Date"])

# Extract basic cyclical date-based features
d_df_nn["DayOfWeek"] = d_df_nn["Date"].dt.dayofweek
d_df_nn["Month"] = d_df_nn["Date"].dt.month

# Cyclical encoding for date-based features
d_df_nn["DayOfWeek_Sin"] = np.sin(2 * np.pi * d_df_nn["DayOfWeek"] / 7)
d_df_nn["DayOfWeek_Cos"] = np.cos(2 * np.pi * d_df_nn["DayOfWeek"] / 7)
d_df_nn["Month_Sin"] = np.sin(2 * np.pi * d_df_nn["Month"] / 12)
d_df_nn["Month_Cos"] = np.cos(2 * np.pi * d_df_nn["Month"] / 12)

# Create lag features
d_df_nn["Lag_1"] = d_df_nn["MinMaxScaled"].shift(1)
d_df_nn["Lag_7"] = d_df_nn["MinMaxScaled"].shift(7)
d_df_nn["Lag_14"] = d_df_nn["MinMaxScaled"].shift(14)

# Create rolling window statistics
d_df_nn["Rolling_7"] = d_df_nn["MinMaxScaled"].rolling(window=7).mean()
d_df_nn["Rolling_7_Std"] = d_df_nn["MinMaxScaled"].rolling(window=7).std()

# Drop NaN values after feature creation
d_df_nn = d_df_nn.dropna().reset_index(drop=True)

# Drop unnecessary categorical columns that NN models don't need
d_df_nn = d_df_nn.drop(columns=["DayOfWeek", "Month"])

# Drop NaN values (if any)
d_df_nn = d_df_nn.dropna().reset_index(drop=True)
######################################################################################################################################
# ML
# Define Features (X) and Target (y)
features = ["Lag_1", "Lag_7", "Lag_14", "Lag_30", "Rolling_7", "Rolling_7_Std",
            "DayOfWeek_Sin", "DayOfWeek_Cos", "Month_Sin", "Month_Cos", "Lag_1_to_Lag_7_Ratio", "Lag_7_to_Lag_14_Ratio"]

train_size = int(len(d_df_ml) * 0.8)

X_train, X_test = d_df_ml.iloc[:train_size][features], d_df_ml.iloc[train_size:][features]
y_train, y_test = d_df_ml.iloc[:train_size]["ShippedQty"], d_df_ml.iloc[train_size:]["ShippedQty"]


# Convert DataFrame to NumPy array and reshape for ML models
X_train = X_train.to_numpy().reshape(X_train.shape[0], -1)  # Convert to NumPy array before reshaping
X_test = X_test.to_numpy().reshape(X_test.shape[0], -1)    # Convert to NumPy array before reshaping


models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net": ElasticNet(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100),
    "LightGBM": LGBMRegressor(n_estimators=100),
    "CatBoost": CatBoostRegressor(n_estimators=100, verbose=0),
    "Support Vector Regression (SVR)": SVR(kernel="rbf", C=1.0, epsilon=0.1)
}


df_results, df_overfitting, predictions = evaluate_models(models, X_train, X_test, y_train, y_test)

st.subheader("Model Performance & Training and Testing Scores")
tab1, tab2 = st.columns(2)

with tab1:
    st.dataframe(df_results)  # Displays the model performance table
with tab2:
    st.dataframe(df_overfitting)  # Displays the overfitting check table
######################################################################################################################################
# ML Plotting
# Ensure dates are aligned with test set
dates_test = d_df_ml.iloc[train_size:]["Date"]
dates_train = d_df_ml.iloc[:train_size]["Date"]


selected_model_name = st.selectbox("Choose a Model:", list(models.keys()))
selected_model = models[selected_model_name]

# Generate & display predictions only when a model is selected
if selected_model:
    st.subheader(f"📈 Predictions for {selected_model_name}")
    fig = plot_each_model_predictions(y_train, y_test, selected_model, dates_train, dates_test, selected_model_name)

    # Generate future forecast
    st.subheader("🔮 Future Forecast for Next Week")
    # future_fig, monthly_forecast = future_forecast(selected_model)
    future_fig, weekly_forecast = future_forecast(selected_model)

    # Display the future forecast visualization
    st.plotly_chart(future_fig)

    # Display forecasted order quantity
    st.success(f"📊 Recommended Order Quantity for Next Week: **{weekly_forecast} units**")




######################################################################################################################################
# # Emsemble Modeling
#
# base_models = [
#     ("catboost", CatBoostRegressor(n_estimators=100, verbose=0)),  # Handles categorical features well
#     ("lightgbm", LGBMRegressor(n_estimators=100)),  # Efficient boosting for large datasets
#     ("xgboost", XGBRegressor(n_estimators=100)),  # Strong performance for structured data
#     ("ridge", Ridge()),  # Adds linear generalization component
#     ("svr", SVR(kernel="rbf"))  # Captures non-linearity
# ]
# stacking_model = StackingRegressor(estimators=base_models, final_estimator=Ridge())
# blending_model = VotingRegressor(estimators=base_models)
#
# plot_each_model_predictions(y_train, y_test, stacking_model, dates_train, dates_test, "Stacking Regressor")
# plot_each_model_predictions(y_train, y_test, blending_model, dates_train, dates_test, "Blending Regressor")
######################################################################################################################################






