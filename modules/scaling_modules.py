# Imports

# 1. Standard library
# (none)

# 2. Third-party
import numpy as np
import pandas as pd
from .targeted_workflow_modules import separate_cohorts


# Functions


def log2_transformation(df):
    """
    Apply log2 transformation to all values in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing untransformed data

    Returns:
    --------
    pd.DataFrame
        DataFrame containing log2-transformed data
    """
    
    df_log2 = df.apply(lambda x: np.log2(x))
    return(df_log2)



def log_urine_conc_normalization_nan_threshold(df_metabo, df_standard, nan_threshold = 0.2):
    """
    Apply MSTUS normalization for log-transformed data.

    Equivalent to dividing by the sum of values in original space. Only features
    with NaN prevalence below nan_threshold are used to calculate the normalization factor.

    Parameters:
    -----------
    df_metabo : pd.DataFrame
        Log-transformed metabolite data
    df_standard : pd.DataFrame
        Log-transformed internal standard data
    nan_threshold : float, optional
        Maximum allowed NaN prevalence for features used in normalization, by default 0.2

    Returns:
    --------
    pd.DataFrame
        Concentration-normalized log-transformed data with internal standards appended
    """
    row_ls = []
    df = df_metabo.copy()
    
    # Calculate NaN prevalence for each feature
    nan_prevalence = df.isna().mean()
    
    # Keep features with NaN prevalence <= threshold
    valid_features = nan_prevalence[nan_prevalence <= nan_threshold].index.tolist()
    df_filtered = df[valid_features]

    for i in range(len(df.index)):
        # Use only features with nan prevalence < nan_threshold for normalization
        row_filtered = df_filtered.iloc[i, :]
        row = df.iloc[i,:]
        # Calculate log(sum(exp(values)))
        # Convert from log space to original space, sum, then back to log
        log_sum = np.log2(np.sum(np.exp2(row_filtered)))        
        # Normalize by subtracting log_sum (equivalent to division in original space)
        row = row - log_sum
        row_ls.append(row)
    
    # Rejoin IS and metabolites
    df_metabo = pd.DataFrame(row_ls, index=df.index)
    df_metabo.columns = df.columns
    df = pd.concat([df_metabo, df_standard], axis=1)
    df.index.names = ["000_Sample_number"]

    return df



def log_median_scaling(df):
    """
    Scale log-transformed data by subtracting the column median.

    Equivalent to division by median in original space.

    Parameters:
    -----------
    df : pd.DataFrame
        Log-transformed data

    Returns:
    --------
    pd.DataFrame
        Median-scaled log-transformed data
    """
    # In log space, division becomes subtraction
    df_scaled = df.apply(lambda x: x - x.median())
    return df_scaled



def log_mean_scaling(df):
    """
    Scale log-transformed data by subtracting the column mean.

    Equivalent to division by mean in original space.

    Parameters:
    -----------
    df : pd.DataFrame
        Log-transformed data

    Returns:
    --------
    pd.DataFrame
        Mean-scaled log-transformed data
    """
    # In log space, division becomes subtraction
    df_scaled = df.apply(lambda x: x - x.mean())
    return df_scaled



def log_uv_scaling(df):
    """
    Apply unit variance scaling to log-transformed data.

    Parameters:
    -----------
    df : pd.DataFrame
        Log-transformed data

    Returns:
    --------
    pd.DataFrame
        Unit variance-scaled log-transformed data
    """
    # Standard deviation scaling is still division in log space
    uv = lambda x: x / np.std(x)
    df_uv = df.apply(uv)
    return df_uv



def log_pareto_scaling(df):
    """
    Apply Pareto scaling to log-transformed data.

    Parameters:
    -----------
    df : pd.DataFrame
        Log-transformed data

    Returns:
    --------
    pd.DataFrame
        Pareto-scaled log-transformed data
    """
    par = lambda x: x / np.sqrt(np.std(x))
    df_pareto = df.apply(par)
    return df_pareto

