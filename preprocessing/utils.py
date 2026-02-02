import pandas as pd

def remove_selected_columns(df, columns_remove):
    """Remove selected columns from dataframe"""
    if not columns_remove:
        return df
    return df.drop(columns=columns_remove, errors='ignore')

def categorical_numerical(df):
    """Function to find categorical and numerical columns/variables in dataset"""
    num_columns, cat_columns = [], []
    for col in df.columns:
        series = df[col]
        # Skip columns containing sequence-like objects (lists/tuples/arrays), e.g., text vectors
        try:
            has_sequences = series.apply(
                lambda v: isinstance(v, (list, tuple)) or (
                    hasattr(v, 'tolist') and not isinstance(v, (str, bytes))
                )
            ).any()
        except Exception:
            has_sequences = False
        if has_sequences:
            continue

        # Numeric dtype detection
        if pd.api.types.is_numeric_dtype(series):
            num_columns.append(col.strip())
            continue

        # Fallback to categorical by low cardinality or object dtype
        try:
            unique_count = series.nunique(dropna=True)
        except TypeError:
            safe_series = series.apply(
                lambda v: tuple(v) if isinstance(v, list) else (
                    tuple(v.tolist()) if hasattr(v, 'tolist') and not isinstance(v, (str, bytes)) else v
                )
            )
            unique_count = safe_series.nunique(dropna=True)

        if unique_count <= 30 or series.dtype == object:
            cat_columns.append(col.strip())
        else:
            num_columns.append(col.strip())
    return num_columns, cat_columns
