# Automated EDA & Model Sandbox Project Workflow

This document outlines the end-to-end workflow of the Streamlit-based Exploratory Data Analysis (EDA) and Machine Learning application.

## 1. Data Ingestion (Upload)
- **Action:** User uploads a dataset via the sidebar.
- **Supported Formats:** CSV (`.csv`), Excel (`.xlsx`, `.xls`).
- **Initial Processing:**
  - The file is read into a Pandas DataFrame.
  - A copy of the original raw data is stored for comparison.
  - Basic validation checks (empty file, parse errors) are performed.

## 2. Data Preprocessing (Sidebar Tools)
Before analysis, users can clean and refine their data using the sidebar tools. Changes here propagate to all analysis tabs.
- **Column Management:**
  - **Select Columns:** Choose specific columns to keep (useful for high-dimensional data).
  - **Change Data Types:** Manually convert columns to Numeric, Categorical, or Datetime.
- **Data Cleaning:**
  - **Handle Missing Values:** Options to drop rows with missing values or fill them (mean/median/mode/zero/forward-fill/backward-fill).
  - **Remove Duplicates:** Identify and remove duplicate rows.
  - **Outlier Handling:** Detect outliers using IQR or Z-Score and handle them (remove, cap, or replace with mean/median).
- **Filtering:**
  - **Filter Data:** Apply logic (e.g., `Age > 30`) to retain specific subsets of data.
  - **Drop Rows:** Remove specific rows based on conditional logic.

## 3. Automated Recommendations (Sidebar)
- **System Analysis:** The app automatically scans the dataset for quality issues.
- **Insights Provided:**
  - High cardinality in categorical columns.
  - Skewed numerical distributions.
  - Significant missing data.
  - Constant or zero-variance columns.
- **Actionable Advice:** Users receive suggestions on how to fix these issues (e.g., "Consider dropping this column" or "Apply log transformation").

## 4. Exploratory Data Analysis (Main Tabs)
The application provides a comprehensive suite of tabs for deep diving into the data.

### A. General Overview
- **📋 Overview:** Displays the first few rows (head) and dataset dimensions.
- **🔍 Missing Data:** Heatmaps and bar charts showing null value distribution.
- **🏷️ Data Types:** Summary of column types and memory usage.
- **📊 Stats & Visuals:** Descriptive statistics (mean, std, min/max) and quick distribution plots.

### B. Detailed Visualization
- **🔎 Search:** Search functionality to find specific rows based on keyword matching.
- **📈 Feature Dist:** Histograms and Boxplots for numerical columns to understand spread and outliers.
- **🎯 Scatter Plot:** Bivariate analysis to see relationships between two numerical variables (with optional color coding).
- **📊 Categorical Analysis:** Bar charts and frequency counts for categorical fields.
- **🔄 Cat vs Num:** Boxplots and Violin plots to analyze numerical distributions across categories.

### C. Advanced Analysis
- **🔬 Feature Exploration:** Deep dive into individual features with specific metrics.
- **📈 PCA Analysis:** Principal Component Analysis to visualize high-dimensional data in 2D or 3D space (dimensionality reduction).
- **🆚 Compare Data:** Side-by-side comparison of the "Original Raw Data" vs. the currently "Processed Data" to track changes.

## 5. Model Sandbox (Machine Learning)
A dedicated environment to train and test predictive models without writing code.

### A. Configuration
- **Target Selection:** User selects the column they want to predict.
- **Problem Type Detection:** The system automatically detects if it's a **Classification** (categorical target) or **Regression** (numerical target) problem.
- **Test Split:** Slider to adjust the training/testing data split ratio (e.g., 80/20).

### B. Training
The app trains multiple algorithms simultaneously:
- **Classification Models:** Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, SVC, KNN.
- **Regression Models:** Linear Regression, Ridge, Lasso, Random Forest Regressor, Gradient Boosting Regressor.

### C. Evaluation & Best Model
- **Performance Metrics:**
  - *Classification:* Accuracy, F1 Score (Macro).
  - *Regression:* R² Score, Mean Squared Error (MSE).
- **Leaderboard:** A table ranks models by performance.
- **Best Model Highlight:** The top-performing model is automatically identified and highlighted.

### D. Prediction
- **Manual Input:** Users can manually enter feature values into a form to get a real-time prediction.
- **Batch Prediction:** Users can upload a new CSV file (without the target column) to generate predictions for multiple rows at once. Results can be downloaded.

## 6. Data Export
- **Download:** Users can download the fully processed/cleaned dataset as a CSV file.
- **Report:** A summary text report of the dataset stats is generated.
