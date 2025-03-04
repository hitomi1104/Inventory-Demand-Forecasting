# 📈 Demand Forecasting Project

## 👉 Step 1: Aggregate the Data
### **Objective**
Convert raw transactional data into structured time-series format to prepare for forecasting models.

### **Aggregation Levels:**
- **Daily** → Best for short-term forecasting, inventory planning.
- **Weekly** → Useful for mid-term trend analysis.
- **Monthly / Quarterly** → Helps in long-term sales pattern identification.

### **Outcome**
A structured dataset with consistent time intervals for accurate forecasting.

---

## 👉 Step 2: Handle Missing Values
### **Objective**
Identify and fill missing values in the dataset to prevent gaps from affecting model predictions.

### **Methods to Handle Missing Data:**
- **Interpolate missing values** → Fill gaps between existing data points.
- **Forward Fill (ffill)** → Carry forward previous day's sales data.
- **Zero-fill** → Set missing values to 0 if SKU was not sold.
- **Impute using historical trends** → Use rolling averages or seasonality-based imputation.

### **Outcome**
A complete dataset without missing values, ensuring consistency in model inputs.

---

## 👉 Step 3: Detect & Handle Outliers
### **Objective**
Identify and handle extreme values that could distort model performance.

### **How to Handle Outliers:**
- **Keep outliers** if they represent real demand (e.g., promotions, sales spikes).
- **Cap outliers** using Winsorization (limit extreme values to a threshold).
- **Use IQR Method** to detect and remove statistical outliers.

### **Outcome**
A cleaned dataset where extreme values are either treated or accounted for appropriately.

---

## 👉 Step 4: Transform the Data
### **Objective**
Convert non-normally distributed data into a more stable form to improve model accuracy.

### **Transformation Methods:**
- **Log Transformation (`log1p()`)** → Reduces right-skewed distributions.
- **Square Root Transformation** → Less aggressive than log, useful for moderate skewness.
- **Box-Cox Transformation** → Applies a power transformation, works for positive values.
- **Yeo-Johnson Transformation** → Handles both positive and negative values.

### **Outcome**
A dataset with normalized distributions that improve forecasting performance.

---

## 👉 Step 5: Scaling the Data
### **Objective**
Ensure that all features have a consistent range to improve model training stability.

### **Choosing the Right Scaling Method:**
- **MinMaxScaler** → Best for deep learning models (LSTM, RNN) and when features have a known fixed range.
- **StandardScaler** → Ideal for normally distributed data, linear regression, and time-series models (ARIMA, SARIMA).
- **RobustScaler** → Best for datasets with outliers, tree-based models (XGBoost, Random Forest), and robust regression.

### **Outcome**
A properly scaled dataset where all features are optimized for model training.

---

## 👉 Step 6: Feature Engineering
### **Objective**
Enhance data with meaningful features tailored for both **traditional statistical models**, **machine learning models**, and **deep learning models** to improve forecasting accuracy.

### **Prepared Datasets:**
- **`d_df_stat`** → Prepared dataset optimized for **ARIMA/SARIMA (no exogenous variables).**
- **`d_df_ml`** → Feature-enhanced dataset for **ML models (XGBoost, Random Forest, etc.).**
- **`d_df_nn`** → Structured dataset for **Neural Networks (LSTM, Transformers, etc.)**

---

## 👉 Step 7: Train-Test Split (Chronological Order)
### **Objective**
Split the dataset into training and testing sets while preserving the time-based sequence to prevent data leakage.

### **Outcome**
A robust train-test split that respects the temporal structure of the data, preparing the dataset for model training and evaluation.

---

## 👉 Step 8: Traditional Statistical Models (ARIMA, SARIMA, Exponential Smoothing)
### **Objective**
Apply classic time-series forecasting models that rely on past demand patterns.

### **Models Included:**
- **ARIMA (Auto-Regressive Integrated Moving Average)** → Best for non-seasonal data.
- **SARIMA (Seasonal ARIMA)** → Best for data with seasonal patterns.
- **Exponential Smoothing Models (Simple, Holt-Winters Method)** → Best for smoothing and capturing trends & seasonality.

### **Outcome**
A set of statistical forecasting models that serve as a benchmark before applying machine learning techniques.

---

## 👉 Step 9: Machine Learning Models
### **Objective**
Apply supervised machine learning models to capture complex demand patterns.

### **Models Included:**
- **Linear Regression, Ridge, Lasso, Elastic Net**
- **Decision Trees, Random Forest, XGBoost, LightGBM, CatBoost**
- **Support Vector Regression (SVR)**
- **Ensemble Models (Stacking, Blending)**

### **Outcome**
More robust forecasting models leveraging ML techniques.

---

## 👉 Step 10: Neural Network Models
### **Objective**
Implement deep learning models capable of learning long-term dependencies.

### **Models Included:**
- **LSTMs (Long Short-Term Memory Networks)**
- **Transformers (Attention-Based Models)**
- **CNNs (Convolutional Neural Networks for Time-Series)**

### **Outcome**
Advanced AI models that dynamically learn sequential patterns.

---

## 👉 Step 11: Model Evaluation & Comparison
### **Objective**
Assess model performance across statistical, ML, and deep learning approaches.

### **Baseline Models:**
- **For Traditional Models:** ARIMA/SARIMA/Exponential Smoothing as the benchmark.
- **For Machine Learning:** Linear Regression as the simplest baseline.
- **For Neural Networks:** A simple LSTM model without attention mechanisms.

### **Metrics Used:**
- **MAE, RMSE, R² Score**
- **Comparison of forecast accuracy**

### **Outcome**
A comprehensive evaluation to determine the best-performing model for demand forecasting.

---




