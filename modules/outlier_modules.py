# Imports

# 1. Standard library
# (none)

# 2. Third-party
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# Functions

def iForest(X):
    """
    Predict anomaly scores and outlier classification for samples using Isolation Forest.

    Parameters:
    -----------
    X : pd.DataFrame
        Input data (samples x features)

    Returns:
    --------
    tuple
        - a_score : np.ndarray — anomaly scores in range [-0.5, 0.5]; lower = more anomalous
        - outliers : np.ndarray — outlier predictions: -1 = outlier, 1 = inlier
    """
    # train & fit iForest
    clf = IsolationForest(max_samples=len(X), random_state=0)
    clf.fit(X)

    # get anomaly scores
    a_score = clf.decision_function(X)

    # predict outliers
    outliers = clf.predict(X)

    return(a_score, outliers)



def IQR(df, cutoff_factor = 1.5):
    """
    Identify outliers based on IQR for each feature.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data without missing values
    cutoff_factor : float, optional
        Outlier threshold multiplier applied to IQR, by default 1.5

    Returns:
    --------
    list of lists
        Each inner list contains the sample IDs of outliers for the corresponding feature
    """
    outlier_ids = []
    for column in df:
        feature = df[column]
        # calculate interquartile range
        q25_na = np.percentile(feature, 25)
        q50_na = np.percentile(feature, 50)
        q75_na = np.percentile(feature, 75)
        iqr = q75_na - q25_na

        # calculate outlier cutoff
        cutoff = cutoff_factor * iqr
        lower = q25_na - cutoff
        upper = q75_na + cutoff

        # calculate outliers
        outliers = [x for x in feature if x < lower or x > upper]
        out_idx = [idx for idx, x in enumerate(feature) if x < lower or x > upper]

        # get sample ids
        sample_idx = list(df.index)
        outlier_idx = [sample_idx[i] for i in out_idx]

        outlier_ids.append(outlier_idx)

    return(outlier_ids)



def IQR_nan(df, cutoff_factor = 1.5):
    """
    Identify outliers based on IQR for each feature, ignoring missing values.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data, may contain missing values
    cutoff_factor : float, optional
        Outlier threshold multiplier applied to IQR, by default 1.5

    Returns:
    --------
    list of lists
        Each inner list contains the sample IDs of outliers for the corresponding feature
    """
    outlier_ids = []
    for column in df:
        feature = df[column]
        # calculate interquartile range
        q25_na = np.nanpercentile(feature, 25)
        q50_na = np.nanpercentile(feature, 50)
        q75_na = np.nanpercentile(feature, 75)
        iqr = q75_na - q25_na

        # calculate outlier cutoff
        cutoff = cutoff_factor * iqr
        lower = q25_na - cutoff
        upper = q75_na + cutoff

        # calculate outliers
        outliers = [x for x in feature if x < lower or x > upper]
        out_idx = [idx for idx, x in enumerate(feature) if x < lower or x > upper]

        # get sample ids
        sample_idx = list(df.index)
        outlier_idx = [sample_idx[i] for i in out_idx]

        outlier_ids.append(outlier_idx)

    return(outlier_ids)



def IQR_outlier_removal(df, out_labels):
    """
    Replace outlier values with NaN.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data containing outliers
    out_labels : list of lists
        Outlier sample IDs per feature, as returned by IQR or IQR_nan

    Returns:
    --------
    pd.DataFrame
        DataFrame with outlier values set to NaN
    """
    # replace outliers by normal dist around mean of non-outlier values
    for i in range(len(df.columns)):
        column = df.columns[i]

        # replace outlier values
        for j in range(len(out_labels[i])):
            df.loc[out_labels[i][j], column] = np.nan

    return(df)



def gaussian_IQR_outlier_correction(df, out_labels, center, random_state=42, absolute=True):
    """
    Replace outlier values by sampling from a Gaussian distribution fitted to non-outlier values.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data containing outliers (samples x features)
    out_labels : list of lists
        Outlier sample IDs per feature, as returned by IQR or IQR_nan
    center : str
        Center of the replacement Gaussian distribution; 'mean' or 'median'
    random_state : int, optional
        Random seed for the replacement values, by default 42
    absolute : bool, optional
        If True, replacement values are made positive (for data in original space).
        Set to False for log-transformed data, by default True

    Returns:
    --------
    pd.DataFrame
        DataFrame with outlier values replaced by Gaussian-sampled values
    """
    rng = np.random.default_rng(random_state)

    # get mean and sd of non-outliers for each feature
    means = []
    medians = []
    sds = []
    for i in range(len(df.columns)):
        ol_ids = out_labels[i]
        df_no_ol = df.loc[~df.index.isin(ol_ids), df.columns[i]]
        means.append(df_no_ol.mean())
        medians.append(df_no_ol.median())
        sds.append(df_no_ol.std())

    # replace outliers by normal dist around mean of non-outlier values
    for i in range(len(df.columns)):
        column = df.columns[i]
        num_replacements = len(out_labels[i])

        # select center of gaussian distribution
        if center == 'mean':
            new_val = rng.normal(means[i], sds[i], num_replacements)
        elif center == 'median':
            new_val = rng.normal(medians[i], sds[i], num_replacements)
        else:
            new_val = None
            print('No center provided')

        # keep replacement values positive (original space only)
        if absolute and new_val is not None:
            new_val = abs(new_val)

        # replace outlier values
        for j in range(len(out_labels[i])):
            df.loc[out_labels[i][j], column] = new_val[j]

    return(df)


