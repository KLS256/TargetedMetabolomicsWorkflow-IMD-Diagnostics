# Imports
# 1. Standard library
# (none)

# 2. Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.impute import KNNImputer


# Set the Seaborn theme
sns.set_theme(style='white')
palette = sns.color_palette('deep')

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,          # Default text size
    'axes.titlesize': 16,     # Title size
    'axes.labelsize': 14,     # Axis label size
    'xtick.labelsize': 10,    # X-axis tick label size
    'ytick.labelsize': 10,    # Y-axis tick label size
    'legend.fontsize': 12,    # Legend text size
    'figure.titlesize': 18,    # Figure title size
    'axes.titleweight': 'bold', # Make titles bold
    'figure.titleweight': 'bold'
})


# Functions

def sample_batch_map(df_samples, batchlist):
    """
    Map sample IDs to their corresponding batch labels.

    Parameters:
    -----------
    df_samples : pd.DataFrame
        Unprocessed sample data
    batchlist : list
        Batch labels for all samples

    Returns:
    --------
    pd.DataFrame
        Map of sample IDs to batch labels
    """
    idx = [x.split('_')[0] if x.split('_')[0] != 'M4D' else x for x in df_samples.columns]
    idx = [x.replace('pos','') if 'pos' in x else x.replace('neg','') if 'neg' in x else x for x in df_samples.columns]

    df_batches = pd.DataFrame(batchlist, index=idx, columns= ['Batches'])

    return(df_batches)



def qc_batch_map(df_merged):
    """
    Retrieve QC samples and map them to their corresponding batches.

    Parameters:
    -----------
    df_merged : pd.DataFrame
        Sample data of merged positive/negative ionization modes

    Returns:
    --------
    pd.DataFrame
        QC samples with batch assignments, sorted by batch
    """
    qc_samples = df_merged.loc[df_merged.index.str.contains('QC')]
    # Generate batch column
    batches = [x.split('_')[2] for x in qc_samples.index]
    qc_samples['Batch'] = batches
    # Sort QC by batches
    qc_samples = qc_samples.sort_values(by=['Batch'],ascending=True)

    return(qc_samples)



def qc_nan_per_batch(df, sampletype, output_path):
    """
    Plot NaN fraction per batch for QC samples.

    Parameters:
    -----------
    df : pd.DataFrame
        QC data with a 'Batch' column
    sampletype : str
        Sample type label used in plot title and filename
    output_path : str
        Directory path where the plot will be saved

    Returns:
    --------
    pd.DataFrame
        DataFrame with batch labels and corresponding NaN fractions
    """
    batch_set = list(set(df['Batch']))
    batch_set.sort()

    # for each batch get NaN Statistics
    nan_batch = []
    for i in batch_set:
        sub_df = df.loc[df['Batch']==i]
        nan_sub = sub_df.isna().sum(axis = 1).sum()
        # get nan fractions
        nan_sub = nan_sub/(sub_df.shape[0] * sub_df.shape[1])
        nan_batch.append(nan_sub)


    nan_data = {'Batch': batch_set, 'Count': nan_batch}
    batch_nan = pd.DataFrame(nan_data)

    g = sns.lineplot(x=batch_nan['Batch'], y=batch_nan['Count'])

    # Create combined labels with sampletype prefix
    g.tick_params(axis='x', rotation=90, labelsize=8)
    g.tick_params(axis='y', labelsize=8)
    g.set_title('QC: NaN per Batch (' + str(sampletype) + ')')
    g.set_ylabel('NaN-Fraction')
    g.set_ylim(0.1, 0.7)
    plt.tight_layout()
    g.figure.savefig(output_path + '/qc_nan_plot_' + str(sampletype) + '.png')
    plt.close()
    return(batch_nan)



def nan_batch_stats(batch_nan):
    """
    Print NaN statistics for batches and flag problematic ones.

    Parameters:
    -----------
    batch_nan : pd.DataFrame
        DataFrame with 'Batch' and 'Count' columns, as returned by
        qc_nan_per_batch or similar functions
    """
    # Report batches with nan above 0.4
    bad_batches = batch_nan[batch_nan['Count'] > 0.4]
    print('The following batches contain more than 40% NaN:')
    print(list(bad_batches['Batch']))

    # Report average nan
    print('Average NaN prevalence: ' + str(round(batch_nan['Count'].mean(), 2)))

    # Recommend changing column if nan prevalence in last batch was > 0.4
    if batch_nan.loc[batch_nan.index[-1], 'Count'] > 0.4:
        print('WARNING!')
        print('The most recent batch contains ' + str(round(batch_nan.loc[batch_nan.index[-1], 'Count'], 2)) + ' NaN.')
        print('Consider changing the column or adapting the processing method.')
        print('')
    


def IS_nan_per_batch(df, sampletype, output_path):
    """
    Plot NaN fraction per batch for internal standard samples.

    Parameters:
    -----------
    df : pd.DataFrame
        Internal standard data
    sampletype : str
        Sample type label used in plot title and filename
    output_path : str
        Directory path where the plot will be saved

    Returns:
    --------
    pd.DataFrame
        DataFrame with batch labels and corresponding NaN fractions
    """
    # Generate batch column
    batches = [x.split('_')[1] for x in df.index]
    df['Batch'] = batches
    # Sort QC by batches
    df = df.sort_values(by=['Batch'],ascending=True)

    batch_set = list(set(df['Batch']))
    batch_set.sort()

    # for each batch get NaN Statistics
    nan_batch = []
    for i in batch_set:
        sub_df = df.loc[df['Batch']==i]
        nan_sub = sub_df.isna().sum(axis = 1).sum()
        # get nan fractions
        nan_sub = nan_sub/(sub_df.shape[0] * sub_df.shape[1])
        nan_batch.append(nan_sub)


    nan_data = {'Batch': batch_set, 'Count': nan_batch}
    batch_nan = pd.DataFrame(nan_data)

    g = sns.lineplot(x=batch_nan['Batch'], y=batch_nan['Count'])
    g.tick_params(rotation=90)
    g.set_title('IS: NaN per Batch (' + str(sampletype) + ')')
    g.set_ylabel('NaN-Fraction')
    g.set_ylim(0.1, 0.7)

    g.figure.savefig(output_path + '/IS_nan_plot_' + str(sampletype) + '.png')
    plt.close()
    return(batch_nan)



def feature_nan_per_batch(df, sampletype, output_path):
    """
    Plot NaN fraction per batch for feature data.

    Parameters:
    -----------
    df : pd.DataFrame
        Sample feature data
    sampletype : str
        Sample type label used in plot title and filename
    output_path : str
        Directory path where the plot will be saved

    Returns:
    --------
    pd.DataFrame
        DataFrame with batch labels and corresponding NaN fractions
    """
    # Generate batch column
    batches = [x.split('_')[1] for x in df.index]
    df['Batch'] = batches
    # Sort QC by batches
    df = df.sort_values(by=['Batch'],ascending=True)
    batch_set = list(set(df['Batch']))
    batch_set.sort()

    # for each batch get NaN Statistics
    nan_batch = []
    for i in batch_set:
        sub_df = df.loc[df['Batch']==i]
        nan_sub = sub_df.isna().sum(axis = 1).sum()
        # get nan fractions
        nan_sub = nan_sub/(sub_df.shape[0] * sub_df.shape[1])
        nan_batch.append(nan_sub)


    nan_data = {'Batch': batch_set, 'Count': nan_batch}
    batch_nan = pd.DataFrame(nan_data)

    g = sns.lineplot(x=batch_nan['Batch'], y=batch_nan['Count'])
    g.tick_params(rotation=90)
    g.set_title('Features: NaN per Batch (' + str(sampletype) + ')')
    g.set_ylabel('NaN-Fraction')
    g.set_ylim(0.1, 0.7)
    plt.tight_layout()
    g.figure.savefig(output_path + '/feature_nan_plot_' + str(sampletype) + '.png')
    plt.close()
    return(batch_nan)



def visualize_missing_data(df, figsize=(16, 8), save_path=None, random_state=42):
    """
    Visualize missing data patterns across features and samples.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to analyze for missing values
    figsize : tuple, optional
        Figure size as (width, height), by default (16, 8)
    save_path : str, optional
        Directory path where plots will be saved. If None, plots are not saved.
    random_state : int, optional
        Random seed for selecting the samples shown in the heatmap, by default 42
    """
    rng = np.random.default_rng(random_state)

    # Calculate missing values by column
    percent_missing_by_column = (df.isna().sum() / df.shape[0] * 100).sort_values(ascending=False)
    # Calculate missing values by row
    missing_by_row = df.isna().sum(axis=1)
    percent_missing_by_row = (missing_by_row / df.shape[1] * 100)
    # Create figure with subplots - 1x3 layout
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Histogram of missing values by sample (increased bins)
    deep_colors = sns.color_palette("deep")
    ax1 = axes[0]
    sns.histplot(percent_missing_by_row, bins=40, kde=False, ax=ax1, color=deep_colors[0])
    ax1.set_title('Distribution of Missing Values Across Samples', fontsize=18)
    ax1.set_xlabel('Percentage of Missing Values')
    ax1.set_ylabel('Count of Samples')
    # Add panel label
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontweight='bold', va='top')
    
    # 2. Heatmap ordered by missingness percentage
    ax2 = axes[1]
    
    # Sample rows if there are too many
    max_samples = 100
    if df.shape[0] > max_samples:
        sample_indices = rng.choice(df.shape[0], max_samples, replace=False)
        sample_df = df.iloc[sample_indices]
    else:
        sample_df = df
    
    # Create binary missing data matrix
    binary_missing = sample_df.isna().astype(int)
    
    # Calculate missingness percentage for features (columns) and samples (rows)
    feature_missingness = binary_missing.mean(axis=0) * 100  # % missing per feature

    # Order features by missingness (ascending - increasing left to right)
    feature_order = feature_missingness.sort_values(ascending=True).index

    # Reorder the heatmap (only reorder columns/features, keep sample order as-is)
    binary_missing_ordered = binary_missing[feature_order]

    greyscale_friendly_cmap = ListedColormap([deep_colors[6], deep_colors[4]])
    # Plot heatmap with bright blue-red colormap
    sns.heatmap(binary_missing_ordered, cbar=False, cmap=greyscale_friendly_cmap, ax=ax2,
                xticklabels=False, yticklabels=False, vmin=0, vmax=1)
    
    # Add vertical line at 60% missingness
    # Find where 60% threshold is in the ordered features
    feature_missingness_ordered = feature_missingness[feature_order]
    threshold_idx = (feature_missingness_ordered < 60).sum()  # Count features with <60% missing

    if 0 < threshold_idx < len(feature_order):
        ax2.axvline(x=threshold_idx, color='black', linewidth=3, linestyle='--')
        
       # Add region labels centered vertically in the heatmap
        mar_x = threshold_idx / 2
        mnar_x = threshold_idx + (len(feature_order) - threshold_idx) / 2
        
        ax2.text(mar_x, binary_missing_ordered.shape[0] + 1, 'MAR', ha='center', va='top', fontweight='bold', fontsize=14)
        ax2.text(mnar_x, binary_missing_ordered.shape[0] + 1, 'MNAR', ha='center', va='top', fontweight='bold', fontsize=14)
    
    ax2.set_title('Missing Value Heatmap', fontsize=18)
    ax2.set_xlabel('Features (Ordered by Missingness)')
    ax2.set_ylabel('Samples')
    
    # Add panel label
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontweight='bold', va='top')
    

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Align x-axis labels after tight_layout
    fig.align_xlabels([ax1, ax2])
    # Save figure if requested
    if save_path:
        plt.savefig(save_path + '/nan_overview.png', dpi=600, bbox_inches='tight')
    
    plt.show()
    
    # Additional feature-focused visualizations
    # Feature-level bar chart with more details
    features_with_missing = percent_missing_by_column[percent_missing_by_column > 0]
    
    if len(features_with_missing) > 0:
        plt.figure(figsize=(20, 10))  # Further increased size for better resolution
        ax = sns.barplot(x=features_with_missing.index, 
                         y=features_with_missing.values,
                         palette=sns.color_palette("YlOrRd", n_colors=len(features_with_missing)))
        
        plt.title('Features with Missing Values', fontsize=16)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Percentage Missing', fontsize=14)
        
        # Set small fontsize for x-labels and rotate them
        plt.xticks(rotation=45, ha='right', fontsize=4)  # Slightly increased fontsize
        
        # No value labels on top of bars
        
        plt.tight_layout()
        
        if save_path:
            feature_path = save_path.replace('.png', '_features.png')
            plt.savefig(feature_path + '/nan_feature_overview.png', dpi=1200, bbox_inches='tight')  # Significantly higher DPI
        
        # Also save a vector format for perfect scaling
        if save_path:
            svg_path = save_path.replace('.png', '_features.svg')
            plt.savefig(svg_path + '/nan_feature_overview_vector.png', format='svg', bbox_inches='tight')
            
        plt.show()
    
    # Additional sample-focused visualizations
    # Create boxplot of missing counts per sample by categorical variables if available
    cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    
    if cat_cols and len(cat_cols) > 0:
        plt.figure(figsize=(12, 6 * len(cat_cols)))
        
        for i, col in enumerate(cat_cols[:3]):  # Limit to first 3 categorical variables
            plt.subplot(len(cat_cols[:3]), 1, i+1)
            
            # Create temporary dataframe with categorical variable and missing count
            temp_df = pd.DataFrame({
                'Category': df[col],
                'Missing_Percentage': percent_missing_by_row
            })
            
            # Drop rows where category is missing
            temp_df = temp_df.dropna(subset=['Category'])
            
            # Create boxplot
            sns.boxplot(x='Category', y='Missing_Percentage', data=temp_df, palette='Set3')
            plt.title(f'Distribution of Missing Values by {col}')
            plt.xlabel(col)
            plt.ylabel('Percentage of Missing Values')
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            cat_path = save_path.replace('.png', '_categories.png')
            plt.savefig(cat_path, dpi=300, bbox_inches='tight')
            
        plt.show()



def half_minimum_scatter_per_feature(df, imp_features, random_state=42):
    """
    Generate Gaussian scatter around half-minimum value independently for each feature.
    Assumes data is already log-transformed and normally distributed in log space.

    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe containing log-transformed data
    imp_features : list
        List of column names to impute
    random_state : int, optional
        Random seed for the Gaussian scatter, by default 42

    Returns:
    --------
    df_imputed : pandas DataFrame
        Copy of input dataframe with NaN values imputed
    """
    rng = np.random.default_rng(random_state)

    # Create a copy of the dataframe
    df_imputed = df.copy()
    
    # Calculate global minimum in log space (to use as fallback)
    global_min_log = df[imp_features].min().min()
    
    # Process each feature separately
    for feature in imp_features:
        # Count NaN values in this feature
        nan_mask = df_imputed[feature].isna()
        nan_count = nan_mask.sum()
        
        if nan_count == 0:
            continue  # Skip if no NaN values for this feature
        
        # Get minimum value for this specific feature (already in log space)
        if df_imputed[feature].notna().any():
            min_val_log = df_imputed[feature].min()
        else:
            # If feature contains only NaNs, use global minimum
            min_val_log = global_min_log
        
        # Calculate the half-minimum value in log space
        # In normal space: x/2 = 2^(log2(x) - 1)
        half_min_log = min_val_log - 1
        
        # Calculate SD directly for log space
        # Simpler approach: use a consistent SD for all features in log space
        # since log-transformed data should have more consistent variance
        sd_log = 1/3  # This means 99.7% of values will be within ±1 of the half-minimum in log space
                      # You may need to adjust this based on your data
        
        # Generate Gaussian scatter specific to this feature
        scatter = rng.normal(half_min_log, sd_log, nan_count)
        
        # Assign scatter values to NaN positions - no conversion needed as we're already in log space
        df_imputed.loc[nan_mask, feature] = scatter
    
    return df_imputed



def kNN_imp(df, n):
    """
    Impute missing values using k-nearest neighbors.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing data with missing values
    n : int
        Number of nearest neighbors for imputation

    Returns:
    --------
    pd.DataFrame
        DataFrame with NaN values replaced by KNN-imputed values
    """
    idx = df.index
    colnames = df.columns
    imputer = KNNImputer(n_neighbors=n)
    df_imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed)
    df_imputed.index = idx
    df_imputed.columns = colnames
    return(df_imputed)



def zero_kNN(df, zero_threshold, numneighbors):
    """
    Impute missing values using a two-stage strategy:
    features exceeding the NaN threshold are imputed with half-minimum scatter,
    remaining NaNs are filled using KNN imputation.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing data with missing values
    zero_threshold : float
        NaN fraction above which a feature is considered MNAR and imputed
        with half-minimum scatter
    numneighbors : int
        Number of nearest neighbors for KNN imputation

    Returns:
    --------
    pd.DataFrame
        DataFrame with all NaN values imputed
    """

    # get number of nan per column (target metabolite)
    nan_col = df.isna().sum()
    # get percentage of nan for each column
    nan_per_col = nan_col/len(df)
    # get columns with nan above threshold and fillna(0)
    imp_features = []
    for i in nan_per_col.index:
        if nan_per_col[i] >= zero_threshold:
            imp_features.append(i)

    # impute missing values
    df = half_minimum_scatter_per_feature(df, imp_features)

    # kNN on rest
    df_imputed = kNN_imp(df, numneighbors)

    return(df_imputed) 



def kNN_imputation_separate_pat(df_ctrl_knn, df_pat, zero_threshold, numneighbors):
    """
    Impute missing values in patient samples using controls as reference.

    For each patient sample, combines the imputed control data with the
    patient replicates and applies zero_kNN imputation.

    Parameters:
    -----------
    df_ctrl_knn : pd.DataFrame
        Already-imputed control data used as reference
    df_pat : pd.DataFrame
        Patient data containing missing values
    zero_threshold : float
        NaN fraction above which MNAR imputation is applied
    numneighbors : int
        Number of nearest neighbors for KNN imputation

    Returns:
    --------
    pd.DataFrame
        Patient data with NaN values imputed
    """
    miss_list = []
    rep_list = []
    for sample in df_pat.index:

            # get replicates
            i = sample.split('_')[0]
            df_rep = df_pat.loc[df_pat.index.str.contains(i)]

            # check if Nan have been filled already (because triplicates)
            if i not in rep_list:
                temp_df = pd.concat([df_ctrl_knn, df_rep])
                temp_miss = zero_kNN(temp_df, zero_threshold, numneighbors)
                df_rep = temp_miss.loc[temp_miss.index.str.contains(i)]
                miss_list.append(df_rep)
            rep_list.append(i)

    # transform list of lists to df
    df_pat_new = miss_list[0]
    for i in range(1,len(miss_list)):
        df_pat_new = pd.concat([df_pat_new, miss_list[i]])

    return(df_pat_new)



def half_minimum_point(df):
    """
    Impute missing values as half-minimum value for each feature.
    Assumes data is already log-transformed.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe containing log-transformed data
        
    Returns:
    --------
    df_imputed : pandas DataFrame
        Copy of input dataframe with NaN values imputed
    """
    # Create a copy of the dataframe
    df_imputed = df.copy()
    
    # Calculate global minimum in log space (to use as fallback)
    global_min_log = df.min().min()
    
    # Process each feature separately
    for feature in df.columns:
        # Count NaN values in this feature
        nan_mask = df_imputed[feature].isna()
        nan_count = nan_mask.sum()
        
        if nan_count == 0:
            continue  # Skip if no NaN values for this feature
            
        # Get minimum value for this specific feature (already in log space)
        if df_imputed[feature].notna().any():
            min_val_log = df_imputed[feature].min()
        else:
            # If feature contains only NaNs, use global minimum
            min_val_log = global_min_log
            
        # Calculate the half-minimum value in log space
        # In normal space: x/2 = 2^(log2(x) - 1)
        half_min_log = min_val_log - 1
        
        # Assign half-minimum value to all NaN positions
        df_imputed.loc[nan_mask, feature] = half_min_log
    
    return df_imputed


