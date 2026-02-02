# Code Architecture & Directory Structure

This document provides a technical overview of the project's codebase, explaining the directory structure, file responsibilities, and how the different components interact.

## 📂 Directory Structure

```text
EDA_PROJECT_STREAMLIT/
├── preprocessing/                  # Backend Logic Modules
│   ├── __init__.py                # Package initialization
│   ├── binning.py                 # Numerical binning logic
│   ├── code_export.py             # Generates Python code for download
│   ├── data_type_conversion.py    # Type casting (int, float, datetime)
│   ├── datetime_features.py       # Extracting year, month, day, etc.
│   ├── duplicates.py              # Duplicate row handling
│   ├── encoding.py                # One-Hot and Label encoding
│   ├── feature_selection.py       # Selecting best features
│   ├── feature_transformation.py  # Log, Sqrt, Box-Cox transforms
│   ├── filtering.py               # Row filtering and dropping
│   ├── imputation.py              # Filling missing values
│   ├── missing_values.py          # Analysis of missing data
│   ├── model_sandbox.py           # Machine Learning training & evaluation
│   ├── outliers.py                # Outlier detection (IQR, Z-score)
│   ├── pca_module.py              # Principal Component Analysis
│   ├── recommendations.py         # Automated data quality insights
│   ├── scaling.py                 # StandardScaler, MinMaxScaler
│   ├── text_processing.py         # Basic NLP cleaning
│   └── utils.py                   # Helper functions
│
├── .gitignore                     # Git exclusion rules
├── EDA.py                         # Visualization & Analysis Functions
├── preprocessing_pipeline.py      # Central Orchestrator (Facade Pattern)
├── PROJECT_WORKFLOW.md            # User-facing workflow documentation
├── README.md                      # Project setup and introduction
├── requirements.txt               # Python dependencies
├── streamlit_app.py               # Main Application Entry Point (Frontend)
└── test_imports.py                # Script to verify module dependencies
```

---

## 🏗️ Architecture Overview

The application follows a **Modular Monolith** architecture tailored for Streamlit. It separates the **Presentation Layer** (UI) from the **Logic Layer** (Preprocessing & Analysis).

### 1. Presentation Layer (Frontend)
*   **File:** `streamlit_app.py`
*   **Role:** This is the entry point. It handles:
    *   **UI Rendering:** Sidebar, Tabs, Buttons, Dataframes.
    *   **State Management:** Uses `st.session_state` to persist the dataset (`current_df`) and analysis results across reruns.
    *   **User Interaction:** Captures inputs (sliders, dropdowns) and triggers backend actions.
*   **Key Connection:** It imports `PreprocessingPipeline` from `preprocessing_pipeline.py` and functions from `EDA.py`.

### 2. Orchestration Layer (Middle Tier)
*   **File:** `preprocessing_pipeline.py`
*   **Role:** Acts as a **Facade**. Instead of importing 15+ separate modules into the main app, `streamlit_app.py` instantiates one `PreprocessingPipeline` object.
*   **Mechanism:** This class initializes instances of all modules in the `preprocessing/` folder and exposes them via a unified API (e.g., `pipeline.imputation.fill_missing_values(...)`).

### 3. Logic Layer (Backend)
*   **Folder:** `preprocessing/`
*   **Role:** Contains pure Python functions and classes that perform the actual data manipulation. These modules are generally independent of Streamlit (except for some status printing) and use **Pandas** and **Scikit-Learn**.
*   **File:** `EDA.py`
*   **Role:** Dedicated specifically to generating Plotly/Matplotlib charts and statistical summaries for the main analysis tabs.

---

## 📄 Detailed File Responsibilities

### Root Files

| File | Description |
| :--- | :--- |
| **`streamlit_app.py`** | **The Main Controller.** Sets up the page layout, handles file uploads, manages the sidebar navigation, and calls the appropriate pipeline methods based on user actions. |
| **`preprocessing_pipeline.py`** | **The Bridge.** A wrapper class that imports all modules from `preprocessing/` and makes them accessible as attributes (e.g., `self.imputation`). |
| **`EDA.py`** | **Visualization Library.** Contains functions like `display_dataset_overview`, `plot_correlation_heatmap`, and `plot_feature_distribution` used in the main tabs. |
| **`requirements.txt`** | **Dependencies.** Lists libraries like `streamlit`, `pandas`, `scikit-learn`, `plotly`, `seaborn`. |

### Preprocessing Modules (`preprocessing/`)

These modules handle specific data science tasks:

| Module | Functionality |
| :--- | :--- |
| **`model_sandbox.py`** | **ML Engine.** Contains logic to train models (RandomForest, LinearReg, etc.), calculate metrics (Accuracy, R2), and select the best model. |
| **`recommendations.py`** | **Advisor.** Scans the dataframe and returns a list of dictionaries suggesting actions (e.g., "Drop column X because it has 90% missing values"). |
| **`imputation.py`** | **Cleaner.** Functions to fill `NaN` values using Mean, Median, Mode, or constants. |
| **`encoding.py`** | **Translator.** Converts categorical text data into numbers using One-Hot or Label Encoding. |
| **`scaling.py`** | **Normalizer.** Standardizes numerical features (StandardScaler, RobustScaler). |
| **`outliers.py`** | **Detector.** Identifies and handles outliers using IQR or Z-Score methods. |
| **`filtering.py`** | **Query Engine.** Logic for the "Filter Data" and "Drop Rows" sidebar tools. |
| **`pca_module.py`** | **Reducer.** Performs Principal Component Analysis for dimensionality reduction and visualization. |
| **`code_export.py`** | **Generator.** (Optional) functionality to generate a Python script reproducing the user's cleaning steps. |

---

## 🔄 Data Flow Example: "Filling Missing Values"

1.  **User Action:** In `streamlit_app.py` (Sidebar), the user selects a column and chooses "Fill with Mean".
2.  **Trigger:** The "Apply" button is clicked.
3.  **Call:** `streamlit_app.py` calls `pipeline.imputation.impute_numerical(...)`.
4.  **Execution:**
    *   The request goes to `preprocessing_pipeline.py`.
    *   It forwards the call to `preprocessing/imputation.py`.
    *   Pandas operations run: `df[col].fillna(df[col].mean())`.
5.  **Return:** The modified DataFrame is returned.
6.  **State Update:** `streamlit_app.py` updates `st.session_state.processed_df` with the new data.
7.  **Rerun:** Streamlit reruns the script, and the main view now shows the data without missing values.
