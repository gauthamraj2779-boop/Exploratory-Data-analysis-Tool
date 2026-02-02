# Project Operations & Functional Capabilities

This document provides a comprehensive catalog of all technical operations, algorithms, and methodologies implemented in the Automated EDA & Model Sandbox tool.

## 1. Data Ingestion & Type Handling
*   **File Parsing:**
    *   **CSV:** Uses `pandas.read_csv`.
    *   **Excel:** Uses `pandas.read_excel` (supports `.xlsx`, `.xls`).
*   **Type Inference:** Automatically detects numerical, categorical, and datetime columns.
*   **Manual Conversion:** Users can force-cast columns to:
    *   `Numeric` (forces coercion to numbers, handles errors).
    *   `Categorical` (converts to string/object).
    *   `Datetime` (parses dates).

## 2. Data Cleaning Operations
### A. Missing Value Imputation
*   **Numeric Strategies:**
    *   `Mean`: Fills with the average.
    *   `Median`: Fills with the middle value (robust to outliers).
    *   `Mode`: Fills with the most frequent value.
    *   `Zero`: Fills with 0.
*   **Categorical Strategies:**
    *   `Mode`: Fills with the most frequent category.
    *   `Constant`: Fills with "Missing" or a custom string.
*   **Row Removal:** Drop rows containing any null values in selected columns.

### B. Outlier Management
*   **Detection Methods:**
    *   **IQR (Interquartile Range):** Identifies values below $Q1 - 1.5 \times IQR$ or above $Q3 + 1.5 \times IQR$.
    *   **Z-Score:** Identifies values with a standard deviation > 3 from the mean.
*   **Handling Actions:**
    *   **Remove:** Deletes the entire row.
    *   **Winsorize (Cap):** Clamps values to the lower/upper bounds.
    *   **Impute:** Replaces outliers with the column Mean or Median.

### C. Duplicate Handling
*   **Detection:** Identifies exact row matches across all columns.
*   **Action:** Removes duplicate rows, keeping the first occurrence.

## 3. Feature Engineering
### A. Scaling & Normalization
*   **Standard Scaler:** Transforms data to have Mean = 0 and Standard Deviation = 1 ($z = \frac{x - \mu}{\sigma}$).
*   **MinMax Scaler:** Scales data to a fixed range [0, 1].
*   **Robust Scaler:** Scales data using statistics that are robust to outliers (IQR).

### B. Categorical Encoding
*   **Label Encoding:** Converts unique string categories to integers (0, 1, 2...). Suitable for ordinal data.
*   **One-Hot Encoding:** Creates binary columns for each category (e.g., `Color_Red`, `Color_Blue`). Suitable for nominal data.

### C. Transformations
*   **Log Transformation:** Applies $log(x+1)$ to reduce skewness in positive data.
*   **Square Root:** Applies $\sqrt{x}$.
*   **Box-Cox:** Power transform to make data more normal-like (requires strictly positive input).

## 4. Exploratory Data Analysis (EDA)
### A. Statistical Analysis
*   **Descriptive Stats:** Count, Mean, Std, Min, 25%, 50%, 75%, Max.
*   **Correlation:** Pearson correlation coefficient matrix to identify linear relationships.
*   **Skewness & Kurtosis:** Measures of distribution asymmetry and "tailedness".

### B. Visualization Engine
*   **Univariate (Single Variable):**
    *   *Numerical:* Histograms (distribution), Boxplots (outliers), Violin Plots (density).
    *   *Categorical:* Bar Charts (frequency), Pie Charts (proportions).
*   **Bivariate (Two Variables):**
    *   *Scatter Plots:* Relationship between two numerics (with hue mapping).
    *   *Box/Violin by Category:* Distribution of a numeric variable grouped by a category.
    *   *Correlation Heatmap:* Color-coded matrix of feature correlations.
*   **Multivariate:**
    *   *PCA (Principal Component Analysis):* Projects high-dimensional data into 2D or 3D space to visualize clusters and variance.

## 5. Machine Learning (Model Sandbox)
The system automatically determines the problem type based on the target variable's data type.

### A. Classification (Categorical Target)
*   **Algorithms:**
    *   **Logistic Regression:** Linear decision boundary.
    *   **Random Forest Classifier:** Ensemble of decision trees (reduces overfitting).
    *   **Decision Tree:** Single tree (interpretable).
    *   **Gradient Boosting (GBM):** Sequential boosting for high accuracy.
    *   **SVC (Support Vector Classifier):** Finds optimal hyperplane.
    *   **KNN (K-Nearest Neighbors):** Instance-based learning.
*   **Metrics:** Accuracy, F1-Score (Macro Average).

### B. Regression (Numerical Target)
*   **Algorithms:**
    *   **Linear Regression:** Basic OLS regression.
    *   **Ridge / Lasso:** Regularized linear regression (L2 / L1 penalty).
    *   **Random Forest Regressor:** Non-linear ensemble.
    *   **Gradient Boosting Regressor:** High-performance boosting.
*   **Metrics:** R² Score (Coefficient of Determination), MSE (Mean Squared Error).

### C. Model Selection
*   **Train/Test Split:** Randomized split (user-defined 10-50% test size).
*   **Evaluation:** All applicable models are trained on the same split.
*   **Winner Selection:**
    *   *Classification:* Highest Accuracy.
    *   *Regression:* Highest R² Score.

## 6. Recommendations Engine
*   **Logic:** Heuristic-based rules engine.
*   **Checks:**
    *   *High Cardinality:* Warns if a categorical column has too many unique values (risk of overfitting).
    *   *High Missingness:* Warns if >20% of data is null.
    *   *Skewness:* Detects highly skewed distributions.
    *   *Constant Columns:* Identifies columns with zero variance.

## 7. Export & Reproducibility
*   **Data Download:** Exports the current `st.session_state.processed_df` to CSV.
*   **Pipeline Code:** Generates a Python script using `scikit-learn` Pipelines that replicates the preprocessing steps applied in the UI.
