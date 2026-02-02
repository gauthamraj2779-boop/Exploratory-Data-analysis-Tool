import pandas as pd

def detect_duplicates(df):
    try:
        duplicates = df.duplicated()
    except TypeError:
        df_conv = df.copy()
        for c in df_conv.columns:
            if df_conv[c].dtype == 'object':
                df_conv[c] = df_conv[c].apply(
                    lambda v: tuple(v) if isinstance(v, list) else (
                        tuple(v.tolist()) if hasattr(v, 'tolist') and not isinstance(v, (str, bytes)) else v
                    )
                )
        duplicates = df_conv.duplicated()
    duplicate_count = duplicates.sum()
    
    return {
        'total_duplicates': duplicate_count,
        'duplicate_percentage': (duplicate_count / len(df)) * 100,
        'duplicate_indices': df[duplicates].index.tolist()
    }

def remove_duplicates(df, subset=None, keep='first'):
    """Remove duplicate rows"""
    original_length = len(df)
    df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = original_length - len(df_cleaned)
    
    return df_cleaned, removed_count
