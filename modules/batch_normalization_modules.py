### Imports

# 1. Standard library
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# 2. Third-party
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer

# Set the Seaborn theme
sns.set_theme(style='white')
palette = sns.color_palette('deep')

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,          # Default text size
    'axes.titlesize': 16,     # Title size
    'axes.labelsize': 14,     # Axis label size
    'xtick.labelsize': 8,    # X-axis tick label size
    'ytick.labelsize': 8,    # Y-axis tick label size
    'legend.fontsize': 12,    # Legend text size
    'figure.titlesize': 18,    # Figure title size
    'axes.titleweight': 'bold', # Make titles bold
    'figure.titleweight': 'bold'
})

### Functions


def compare_stability_options(data, out_dir, nan_threshold = 0.2, similarity_threshold = 2, plot_results=True):
    """
    Compare the results of normalization with and without stability checking.
    
    Parameters:
    -----------
    data: Input DataFrame
    out_dir: Output directory path where comparison plots will be saved
    nan_threshold: Maximum allowed NaN prevalence for features (default: 0.2)
    similarity_threshold: Maximum allowed fold-change (max/min) between QC replicate
        values for a feature to be considered stable (default: 2)
    plot_results: Whether to generate comparison plots (default: True)
    
    Returns:
    --------
    Dictionary containing comparison results
    """
    
    # Generate synthetic data if not provided
    if data is None:
        print("Please provide input data for batch normalization.")
            
    # Run the workflow with stability checking (default)
    results_with_stability = normalize_batched_data_enhanced(data, nan_threshold, similarity_threshold, check_stability=True)
    
    # Run the workflow without stability checking
    results_without_stability = normalize_batched_data_enhanced(data, nan_threshold, similarity_threshold, check_stability=False)
    
    # Extract the normalization coefficients
    coeff_with_stability = results_with_stability['normalization_coefficients']
    coeff_without_stability = results_without_stability['normalization_coefficients']
    
    # Find batches that appear in both results
    common_batches = set(coeff_with_stability.index) & set(coeff_without_stability.index)
    
    # Compare coefficient values for common batches
    if common_batches:
        comparison_df = pd.DataFrame({
            'with_stability': coeff_with_stability.loc[list(common_batches), 'normalization_coefficient'],
            'without_stability': coeff_without_stability.loc[list(common_batches), 'normalization_coefficient']
        })
        comparison_df['ratio'] = comparison_df['with_stability'] / comparison_df['without_stability']
        
    # Apply normalization to the data using both coefficient sets
    normalized_with_stability = apply_normalization(data, results_with_stability)
    normalized_without_stability = apply_normalization(data, results_without_stability)
    
    # Compare feature statistics before and after normalization
    feature_stats = defaultdict(dict)
        
    # Calculate statistics for original data
    for batch in common_batches:
        batch_samples = [idx for idx in data.index if f'{batch}' in idx]
        if batch_samples:
            batch_data = data.loc[batch_samples]
            feature_stats[batch]['original_mean'] = batch_data.mean().mean()
            feature_stats[batch]['original_std'] = batch_data.std().mean()
    
    # Calculate statistics for normalized data (with stability)
    for batch in common_batches:
        batch_samples = [idx for idx in normalized_with_stability.index if f'{batch}' in idx]
        if batch_samples:
            batch_data = normalized_with_stability.loc[batch_samples]
            feature_stats[batch]['stable_mean'] = batch_data.mean().mean()
            feature_stats[batch]['stable_std'] = batch_data.std().mean()
    
    # Calculate statistics for normalized data (without stability)
    for batch in common_batches:
        batch_samples = [idx for idx in normalized_without_stability.index if f'{batch}' in idx]
        if batch_samples:
            batch_data = normalized_without_stability.loc[batch_samples]
            feature_stats[batch]['unstable_mean'] = batch_data.mean().mean()
            feature_stats[batch]['unstable_std'] = batch_data.std().mean()

    # Convert to DataFrame for easier analysis
    stats_df = pd.DataFrame.from_dict(feature_stats, orient='index')
    
    # Sort dfs by index
    stats_df = stats_df.sort_index()
    comparison_df = comparison_df.sort_index()
    
    # Plot comparison if requested
    if plot_results and common_batches:
        fig = plt.figure(figsize=(12, 8))

        # Plot 1: Coefficient comparison
        plt.subplot(2, 2, 1)
        comparison_df[['with_stability', 'without_stability']].plot(kind='bar', ax=plt.gca(), color=palette)
        plt.title('Normalization Coefficients by Batch')
        plt.ylabel('Coefficient Value')
        plt.xlabel('Batch')
        plt.legend(['With Stability', 'Without Stability'])
        
        # Plot 2: Batch means (original)
        plt.subplot(2, 2, 2)
        stats_df['original_mean'].plot(kind='bar', ax=plt.gca(), color=palette[4])
        plt.title('Batch Means (Original Data)')
        plt.ylabel('Mean Value')
        plt.xlabel('Batch')
        
        # Plot 3: Batch means (with stability)
        plt.subplot(2, 2, 3)
        stats_df['stable_mean'].plot(kind='bar', ax=plt.gca(), color=palette[0])
        plt.title('Normalized Batch Means (With Stability)')
        plt.ylabel('Mean Value')
        plt.xlabel('Batch')
        
        # Plot 4: Batch means (without stability)
        plt.subplot(2, 2, 4)
        stats_df['unstable_mean'].plot(kind='bar', ax=plt.gca(), color=palette[1])
        plt.title('Normalized Batch Means (Without Stability)')
        plt.ylabel('Mean Value')
        plt.xlabel('Batch')
        plt.tight_layout()
        fig.savefig(fname=os.path.join(out_dir, 'stability_comparison.png'))        
    
    return {
        'with_stability': results_with_stability,
        'without_stability': results_without_stability,
        'feature_stats': stats_df,
        'coefficient_comparison': comparison_df if common_batches else None
    }



def plot_coefficient_comparison(comparison_df, out_dir, y_padding=0.3):
    """
    Generate a detailed plot comparing normalization coefficients.
    
    Parameters:
    -----------
    comparison_df: DataFrame containing coefficient comparison data
    y_padding: Factor to increase y-axis limits (default: 0.3 or 30% extra space)
    """

    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar chart of coefficients by batch
    comparison_df[['with_stability', 'without_stability']].plot(
        kind='bar', 
        ax=ax1,
        color=[palette[0], palette[1]]
    )
    ax1.set_title('Normalization Coefficients by Batch')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_xlabel('Batch')
    ax1.legend(['With Stability', 'Without Stability'])
    
    # Increase y-axis limits to make room for labels
    y_max = comparison_df[['with_stability', 'without_stability']].values.max()
    ax1.set_ylim(0, y_max * (1 + y_padding))
    
    # Plot 2: Ratio of coefficients (with/without stability)
    comparison_df['ratio'].plot(kind='bar', ax=ax2, color=palette[2])
    ax2.set_title('Ratio of Coefficients (With Stability / Without Stability)')
    ax2.set_ylabel('Ratio')
    ax2.set_xlabel('Batch')
    ax2.axhline(y=1.0, color=palette[3], linestyle='--', alpha=0.7)
    
    # Add ratio values on top of bars
    for container in ax2.containers:
        labels = []
        for v in container:
            height = v.get_height()
            if height > 1.1 or height < 0.9:
                labels.append(f'{height:.2f}')
            else:
                labels.append('')  # Empty string for values between 0.9 and 1.1
        ax2.bar_label(container, labels=labels)
    
    # Increase y-axis limits to make room for labels
    ratio_max = comparison_df['ratio'].max()
    ax2.set_ylim(0, ratio_max * (1 + y_padding))
    
    # Add a text box with summary statistics
    # First, create a dummy legend to get its style properties
    legend = ax2.legend()
    legend_props = legend.get_frame()
    legend.remove()  # Remove the dummy legend

    # Now apply those properties to your text box
    stats_text = (
        f"Mean Ratio: {comparison_df['ratio'].mean():.2f}\n"
        f"Min Ratio: {comparison_df['ratio'].min():.2f}\n"
        f"Max Ratio: {comparison_df['ratio'].max():.2f}\n"
        f"CV: {comparison_df['ratio'].std() / comparison_df['ratio'].mean():.2f}"
    )

    # Create text box with legend-like properties
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(
                boxstyle='round',  # Legends typically have rounded corners
                facecolor=legend_props.get_facecolor(),
                edgecolor=legend_props.get_edgecolor(),
                alpha=legend_props.get_alpha(),
                linewidth=legend_props.get_linewidth(),
            ))
    plt.tight_layout()
    plt.savefig(out_dir + '/stability_coefficient_ratios.png')



def normalize_batched_data_enhanced(data: pd.DataFrame, nan_threshold: float = 0.2, 
                                   similarity_threshold = 2, check_stability: bool = False) -> Dict[str, any]:
    """
    Enhanced function to normalize batched data with configurable parameters.
    
    Parameters:
    -----------
    data: Input DataFrame
    nan_threshold: Maximum allowed NaN prevalence for features (default: 0.2)
    similarity_threshold: Maximum allowed fold-change (max/min) between QC replicate 
    values for a feature to be considered stable and included in coefficient 
    calculation (default: 2)
    check_stability: Whether to check for stable features (default: True)
    
    Returns:
    --------
    Dictionary containing:
    - 'qc_samples_raw': DataFrame with QC samples before NaN handling
    - 'qc_samples': DataFrame with QC samples after NaN handling
    - 'normalization_coefficients': DataFrame with normalization coefficients
    - 'imputed_batches': List of batch names that were imputed
    """
    # Step 1: Extract QC samples
    qc_samples_raw = extract_qc_samples_only(data)
    
    # Step 2: Handle NaN values in QC samples
    qc_samples = handle_nan_values(qc_samples_raw, nan_threshold)
    
    # Step 3: Get normalization coefficients
    normalization_df = get_normalization_coefficients_flexible(qc_samples, similarity_threshold, check_stability)
    
    # Step 4: Identify missing QCs
    missing_qc_batches = identify_missing_qc_batches(data, normalization_df)
    
    # Step 5: Impute coefficients for missing QC batches
    imputed_batches = []
    if missing_qc_batches:
        normalization_df, imputed_batches = impute_normalization_coefficients(normalization_df, missing_qc_batches)
    
    # Return all relevant dataframes and information
    return {
        'qc_samples_raw': qc_samples_raw,
        'qc_samples': qc_samples,
        'normalization_coefficients': normalization_df,
        'imputed_batches': imputed_batches
    }




def normalize_data_with_coefficients(data: pd.DataFrame, norm_coeffs: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the data using the provided normalization coefficients.
    
    For each sample, this function:
    1. Extracts the batch number from the sample index
    2. Retrieves the corresponding normalization coefficient
    3. Divides the sample's features by this coefficient
    
    Parameters:
    -----------
    data: Original input DataFrame
    norm_coeffs: DataFrame with normalization coefficients (index = batch number, column = 'normalization_coefficient')
    
    Returns:
    --------
    DataFrame with normalized data
    """
    # Create a copy of the input data to avoid modifying the original
    normalized_data = data.copy()
    
    # Process each sample
    for sample_idx in normalized_data.index:
        try:
            # Extract batch number from sample index
            batch = extract_batch_from_index(sample_idx)
            
            # Get normalization coefficient for this batch
            if batch in norm_coeffs.index:
                coefficient = norm_coeffs.loc[batch, 'normalization_coefficient']
                
                # Normalize the sample's features by dividing by the coefficient
                normalized_data.loc[sample_idx] = normalized_data.loc[sample_idx] / coefficient
            else:
                print(f"Warning: No normalization coefficient found for batch {batch}")
        
        except ValueError:
            print(f"Warning: Could not extract batch number from index: {sample_idx}")
    
    return normalized_data




def normalize_with_all_samples(data: pd.DataFrame, nan_threshold: float = 0.2, 
                              check_stability: bool = True) -> Dict[str, any]:
    """
    Normalize batched data using all samples instead of just QC samples.
    
    This function:
    1. Uses all samples in each batch (not just QC samples)
    2. Handles NaN values in samples
    3. Calculates normalization coefficients
    4. Identifies and handles any missing batches
    
    Parameters:
    -----------
    data: Input DataFrame
    nan_threshold: Maximum allowed NaN prevalence for features (default: 0.2)
    check_stability: Whether to check for stable features (default: True)
    
    Returns:
    --------
    Dictionary containing:
    - 'samples_raw': DataFrame with all samples before NaN handling
    - 'samples_processed': DataFrame with all samples after NaN handling
    - 'normalization_coefficients': DataFrame with normalization coefficients
    - 'imputed_batches': List of batch names that were imputed (unlikely to have any)
    """
    # Step 1: Make a copy of all samples
    samples_raw = data.copy()
    
    # Step 2: Handle NaN values in samples
    samples_processed = handle_nan_values(samples_raw, nan_threshold)
    
    # Step 3: Extract batch information and group samples by batch
    batch_groups = {}
    for idx in samples_processed.index:
        try:
            batch = extract_batch_from_index(idx)
            if batch not in batch_groups:
                batch_groups[batch] = []
            batch_groups[batch].append(idx)
        except ValueError:
            continue
    
    # Step 4: Calculate normalization coefficients
    results = {}
    
    for batch, sample_indices in batch_groups.items():
        batch_samples = samples_processed.loc[sample_indices]
        
        # When checking stability, we need at least 2 samples
        if check_stability and len(batch_samples) < 2:
            continue
        
        # Determine which features to use
        if check_stability:
            # For each feature, check if values are similar across samples (within factor of 2)
            similar_features = []
            
            for feature in batch_samples.columns:
                feature_values = batch_samples[feature]
                
                # Skip features with any NaN values
                if feature_values.isna().any():
                    continue
                
                # Check if values are within factor of 2
                min_val = feature_values.min()
                max_val = feature_values.max()
                
                # Handle case where min_val is zero or very close to zero
                if min_val <= 1e-10:
                    continue
                    
                if max_val / min_val <= 2.0:
                    similar_features.append(feature)
            
            # Use only similar features if any were found
            if similar_features:
                features_to_use = similar_features
            else:
                continue  # Skip this batch if no similar features found
        else:
            # Use all features when not checking stability
            features_to_use = batch_samples.columns.tolist()
        
        # Calculate the mean for each feature to use
        feature_means = batch_samples[features_to_use].mean()
        
        # Sum the means across features
        sum_of_means = feature_means.sum()

        # Get mean of sum of means
        sum_of_means = sum_of_means/len(feature_means)
        
        # Store result
        results[batch] = sum_of_means
    
    # Convert results to DataFrame
    normalization_df = pd.DataFrame({'normalization_coefficient': results})
    
    # Step 5: Check if any batches are missing from our results
    all_batches = set(batch_groups.keys())
    batches_with_coeffs = set(normalization_df.index)
    
    missing_batches = all_batches - batches_with_coeffs
    
    # Step 6: Impute coefficients for missing batches (unlikely, but just in case)
    imputed_batches = []
    if missing_batches:
        normalization_df, imputed_batches = impute_normalization_coefficients(
            normalization_df, list(missing_batches))
    
    # Return all relevant dataframes and information
    return {
        'samples_raw': samples_raw,
        'samples_processed': samples_processed,
        'normalization_coefficients': normalization_df,
        'imputed_batches': imputed_batches
    }



def compare_normalization_methods(original_data, out_dir, normalized_df1, normalized_df2, 
                                norm_coeffs1=None, norm_coeffs2=None,
                                method1_name="QC Normalization", 
                                method2_name="Sample Normalization",
                                plot_results=True):
    """
    Compare two different normalization methods.
    
    Parameters:
    -----------
    original_data: Original unnormalized data
    normalized_df1: Data normalized with first method
    normalized_df2: Data normalized with second method
    norm_coeffs1: Normalization coefficients for first method (optional)
    norm_coeffs2: Normalization coefficients for second method (optional)
    method1_name: Name of first method for labeling
    method2_name: Name of second method for labeling
    plot_results: Whether to generate comparison plots
    
    Returns:
    --------
    Dictionary containing comparison results
    """
    
    # Compare coefficients if provided
    if norm_coeffs1 is not None and norm_coeffs2 is not None:
        compare_coefficients = True
        
        # Ensure the norm_coeffs are DataFrames with the correct structure
        if not isinstance(norm_coeffs1, pd.DataFrame) or 'normalization_coefficient' not in norm_coeffs1.columns:
            norm_coeffs1 = pd.DataFrame({'normalization_coefficient': norm_coeffs1})
            
        if not isinstance(norm_coeffs2, pd.DataFrame) or 'normalization_coefficient' not in norm_coeffs2.columns:
            norm_coeffs2 = pd.DataFrame({'normalization_coefficient': norm_coeffs2})
        
        # Find batches that appear in both results
        common_batches = set(norm_coeffs1.index) & set(norm_coeffs2.index)
        
        # Compare coefficient values for common batches
        if common_batches:
            comparison_df = pd.DataFrame({
                method1_name: norm_coeffs1.loc[list(common_batches), 'normalization_coefficient'],
                method2_name: norm_coeffs2.loc[list(common_batches), 'normalization_coefficient']
            })
            # Sort the comparison DataFrame by index
            comparison_df = comparison_df.sort_index()
            
            comparison_df['ratio'] = comparison_df[method1_name] / comparison_df[method2_name]
           
    else:
        compare_coefficients = False
        comparison_df = None
    
    # Extract unique batch numbers from the data
    all_batches = set()
    for idx in original_data.index:
        try:
            batch = extract_batch_from_index(idx)
            all_batches.add(batch)
        except ValueError:
            continue
    
    if not all_batches:
        print("WARNING: Could not extract any batch numbers from the data indices.")
        print("Sample index format:", original_data.index[0] if len(original_data.index) > 0 else "No samples")
        return {'error': 'No batch information found in data indices'}
        
    # Compare feature statistics before and after normalization
    feature_stats = defaultdict(dict)
    
    # Calculate statistics for original data
    for batch in all_batches:
        batch_samples = [idx for idx in original_data.index if f'{batch}' in idx]
        if batch_samples:
            batch_data = original_data.loc[batch_samples]
            feature_stats[batch]['original_mean'] = batch_data.mean().mean()
            feature_stats[batch]['original_std'] = batch_data.std().mean()
    
    # Calculate statistics for first normalized data
    for batch in all_batches:
        batch_samples = [idx for idx in normalized_df1.index if f'{batch}' in idx]
        if batch_samples:
            batch_data = normalized_df1.loc[batch_samples]
            feature_stats[batch][f'{method1_name}_mean'] = batch_data.mean().mean()
            feature_stats[batch][f'{method1_name}_std'] = batch_data.std().mean()
    
    # Calculate statistics for second normalized data
    for batch in all_batches:
        batch_samples = [idx for idx in normalized_df2.index if f'{batch}' in idx]
        if batch_samples:
            batch_data = normalized_df2.loc[batch_samples]
            feature_stats[batch][f'{method2_name}_mean'] = batch_data.mean().mean()
            feature_stats[batch][f'{method2_name}_std'] = batch_data.std().mean()
    
    # Convert to DataFrame for easier analysis and sort by index
    stats_df = pd.DataFrame.from_dict(feature_stats, orient='index').sort_index()
    
    # Check if we have data
    if stats_df.empty:
        print("WARNING: No batch statistics could be calculated.")
        return {'error': 'No batch statistics available'}
    
    # Check for required columns
    required_cols = ['original_mean', f'{method1_name}_mean', f'{method2_name}_mean']
    missing_cols = [col for col in required_cols if col not in stats_df.columns]
    
    if missing_cols:
        print(f"WARNING: Missing columns in stats_df: {missing_cols}")
        print("Available columns:", stats_df.columns.tolist())
        return {'error': 'Missing required columns in batch statistics', 'stats_df': stats_df}
    
    # Calculate inter-batch variability
    original_cv = (stats_df['original_mean'].std() / stats_df['original_mean'].mean()) * 100
    method1_cv = (stats_df[f'{method1_name}_mean'].std() / stats_df[f'{method1_name}_mean'].mean()) * 100
    method2_cv = (stats_df[f'{method2_name}_mean'].std() / stats_df[f'{method2_name}_mean'].mean()) * 100
    
    # Plot comparison if requested
    if plot_results:
        # Create figure for data comparison
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Batch means (original)
        plt.subplot(2, 2, 1)
        stats_df['original_mean'].plot(kind='bar', ax=plt.gca(), color=palette[2])
        plt.title('Batch Means (Original Data)')
        plt.ylabel('Mean Value')
        plt.xlabel('Batch')
        plt.gca().text(-0.1, 1.05, 'A', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
        
        # Plot 2: Batch means (method 1)
        plt.subplot(2, 2, 2)
        stats_df[f'{method1_name}_mean'].plot(kind='bar', ax=plt.gca(), color=palette[0])
        plt.title(f'Batch Means ({method1_name})')
        plt.ylabel('Mean Value')
        plt.xlabel('Batch')
        plt.gca().text(-0.1, 1.05, 'B', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

        # Plot 3: Batch means (method 2)
        plt.subplot(2, 2, 3)
        stats_df[f'{method2_name}_mean'].plot(kind='bar', ax=plt.gca(), color=palette[1])
        plt.title(f'Batch Means ({method2_name})')
        plt.ylabel('Mean Value')
        plt.xlabel('Batch')
        plt.gca().text(-0.1, 1.05, 'C', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
        
        # Plot 4: Inter-batch coefficient of variation comparison
        plt.subplot(2, 2, 4)
        
        # Calculate coefficient of variation (std/mean) for each method
        original_mean = stats_df['original_mean'].mean()
        original_std = stats_df['original_mean'].std()
        original_cv = (original_std / original_mean) * 100  # as percentage
        
        method1_mean = stats_df[f'{method1_name}_mean'].mean()
        method1_std = stats_df[f'{method1_name}_mean'].std()
        method1_cv = (method1_std / method1_mean) * 100  # as percentage
        
        method2_mean = stats_df[f'{method2_name}_mean'].mean()
        method2_std = stats_df[f'{method2_name}_mean'].std()
        method2_cv = (method2_std / method2_mean) * 100  # as percentage
        
        variability = pd.DataFrame({
            'Coefficient of Variation (%)': [
                original_cv,
                method1_cv,
                method2_cv
            ]
        }, index=['Original', method1_name, method2_name])
        
        # Plot Inter Batch Coefficient of Variation
        ax = plt.gca()
        variability['Coefficient of Variation (%)'].plot(
            kind='bar', 
            ax=ax,
            color=[palette[2], palette[0], palette[1]]
        )
        
        plt.title('Inter-batch Coefficient of Variation')
        plt.ylabel('CV of Batch Means (%)')
        plt.xlabel('Normalization Method')
        plt.xticks(rotation=0)
        # Set x-tick label size manually for this subplot
        plt.gca().tick_params(axis='x', labelsize=12)
        plt.gca().text(-0.1, 1.05, 'D', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
        
        # Add percentage reduction annotations
        method1_reduction = (1 - method1_cv/original_cv) * 100
        method2_reduction = (1 - method2_cv/original_cv) * 100
        
        plt.annotate(f'{method1_reduction:.1f}% reduction', 
                    (1, method1_cv), 
                    xytext=(1, method1_cv + original_cv*0.1),
                    ha='center',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

        plt.annotate(f'{method2_reduction:.1f}% reduction', 
                    (2, method2_cv), 
                    xytext=(2, method2_cv + original_cv*0.1),
                    ha='center',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
                
        plt.tight_layout()
        plt.savefig(out_dir + '/qc_vs_sample_normalization.png', 
            dpi=600)

        # Generate a separate plot for coefficient comparison if coefficients were provided
        if compare_coefficients and comparison_df is not None:
            plot_coefficient_comparison_methods(comparison_df, method1_name, method2_name, out_dir)
    
    # Prepare return data
    result = {
        'feature_stats': stats_df,
        'variability_comparison': {
            'original_cv': original_cv,
            f'{method1_name}_cv': method1_cv,
            f'{method2_name}_cv': method2_cv
        }
    }
    
    # Add coefficient comparison if available
    if compare_coefficients and comparison_df is not None:
        result['coefficient_comparison'] = comparison_df
    
    return result



def extract_qc_samples_only(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all QC samples from the dataset without NaN handling.
    
    Parameters:
    -----------
    data: Input DataFrame with sample indices
    
    Returns:
    --------
    DataFrame containing only QC samples
    """
    # Extract samples with 'QC' in their index
    qc_samples = data[data.index.str.contains('QC')]
    
    if qc_samples.empty:
        raise ValueError("No QC samples found in the data")
        
    return qc_samples



def handle_nan_values(df: pd.DataFrame, nan_threshold: float = 0.2) -> pd.DataFrame:
    """
    Handle NaN values in a DataFrame by:
    1. Removing features with NaN prevalence > threshold
    2. Imputing remaining NaN values using KNN imputer (n=5)
    
    Parameters:
    -----------
    df: Input DataFrame
    nan_threshold: Maximum allowed NaN prevalence for features (default: 0.2)
    
    Returns:
    --------
    DataFrame with NaN values handled
    """
    # Calculate NaN prevalence for each feature
    nan_prevalence = df.isna().mean()
    
    # Keep features with NaN prevalence <= threshold
    valid_features = nan_prevalence[nan_prevalence <= nan_threshold].index.tolist()
    df_filtered = df[valid_features]
    
    # Impute remaining NaN values using KNN imputer
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_filtered),
        index=df_filtered.index,
        columns=df_filtered.columns
    )
    
    return df_imputed



def extract_batch_from_index(index: str) -> str:
    """
    Extract batch number from a sample index using string splitting.
    
    Parameters:
    -----------
    index: Sample index string (ends with *batch_number)
    
    Returns:
    --------
    Batch number (e.g., 'A001')
    """
    if index.split('_')[0] == 'M4D':
        batch = index.split("_")[2]
        return batch
    else:
        try:
            # Split the index at "_" and take the second part
            batch = index.split("_")[1]
            return batch
        except (IndexError, AttributeError):
            raise ValueError(f"Could not extract batch number from index: {index}")



def get_normalization_coefficients_flexible(qc_samples: pd.DataFrame, similarity_threshold = 2,
                                           check_stability: bool = True) -> pd.DataFrame:
    """
    Calculate normalization coefficients with optional stability checking.
    
    Parameters:
    -----------
    qc_samples: DataFrame containing QC samples (with NaN already handled)
    check_stability: Whether to check for stable features (True) or use all features (False)
    similarity_threshold: Maximum allowed fold-change (max/min) between QC replicate 
    values for a feature to be considered stable and included in coefficient 
    calculation (default: 2)

    Returns:
    --------
    DataFrame with normalization coefficients for each batch
    """
    if qc_samples.empty:
        raise ValueError("No QC samples found in the data")
    
    # Create a dictionary to store results
    results = {}
    
    # Get all unique batch numbers from QC samples
    batch_numbers = set()
    for idx in qc_samples.index:
        try:
            batch = extract_batch_from_index(idx)
            batch_numbers.add(batch)
        except ValueError:
            continue
    
    # Process each batch
    for batch in batch_numbers:
        # Get QC samples for this batch
        batch_qc_indices = [idx for idx in qc_samples.index if f'{batch}' in idx]
        batch_qc_samples = qc_samples.loc[batch_qc_indices]
        
        # When checking stability, we need at least 2 QC samples
        if check_stability and len(batch_qc_samples) < 2:
            continue
        
        # Determine which features to use
        if check_stability:
            # For each feature, check if values are similar across QC samples (within factor of 2)
            similar_features = []
            
            for feature in batch_qc_samples.columns:
                feature_values = batch_qc_samples[feature]
                
                # Skip features with any NaN values (should be already handled, but just in case)
                if feature_values.isna().any():
                    continue
                
                # Check if values are within factor of similarity threshold
                min_val = feature_values.min()
                max_val = feature_values.max()
                
                # Handle case where min_val is zero or very close to zero
                if min_val <= 1e-10:
                    continue
                    
                if max_val / min_val <= similarity_threshold:
                    similar_features.append(feature)
            
            # Use only similar features if any were found
            if similar_features:
                features_to_use = similar_features
            else:
                continue  # Skip this batch if no similar features found
        else:
            # Use all features when not checking stability
            features_to_use = batch_qc_samples.columns.tolist()
       
        # Calculate the mean for each feature to use
        feature_means = batch_qc_samples[features_to_use].mean()
        
        # Sum the means across features
        sum_of_means = feature_means.sum()
        
        # Get mean of sum of means
        sum_of_means = sum_of_means/len(feature_means)

        # Store result
        results[batch] = sum_of_means
    
    # Convert results to DataFrame
    normalization_df = pd.DataFrame({'normalization_coefficient': results})
    
    return normalization_df



def identify_missing_qc_batches(data: pd.DataFrame, normalization_df: pd.DataFrame) -> List[str]:
    """
    Identify batches that are present in the data but don't have QC samples.
    
    Parameters:
    -----------
    data: Input DataFrame
    normalization_df: DataFrame with normalization coefficients from QC samples
    
    Returns:
    --------
    List of batch numbers without QC samples
    """
    # Extract all batch numbers from the data
    all_batches = set()
    for idx in data.index:
        try:
            batch = extract_batch_from_index(idx)
            all_batches.add(batch)
        except ValueError:
            continue
    
    # Get batches with normalization coefficients
    qc_batches = set(normalization_df.index)
    
    # Find batches without QC samples
    missing_qc_batches = all_batches - qc_batches
    
    return sorted(list(missing_qc_batches))



def impute_normalization_coefficients(normalization_df: pd.DataFrame, missing_batches: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Impute normalization coefficients for batches with missing QC samples.
    
    For each batch without QC samples, this function:
    - Finds the 5 closest batches (by batch number)
    - Calculates the mean of their coefficients
    
    Parameters:
    -----------
    normalization_df: DataFrame with normalization coefficients
    missing_batches: List of batch numbers without QC samples
    
    Returns:
    --------
    Tuple containing:
    - Updated normalization DataFrame with imputed coefficients
    - List of batch names that were imputed
    """
    # Convert batch numbers to a format that can be compared numerically
    def batch_to_numeric(batch: str) -> int:
        """Convert batch ID like 'A001' to a numeric value"""
        letter = batch[0]
        number = int(batch[1:])
        return (ord(letter) - ord('A')) * 1000 + number
    
    # Create a copy of the normalization DataFrame
    result_df = normalization_df.copy()
    # Convert existing batch indices to numeric for distance calculation
    existing_batches = result_df.index.tolist()
    existing_batch_values = {batch: batch_to_numeric(batch) for batch in existing_batches}
    
    # List to store imputed batch names
    imputed_batches = []
    
    # Process each missing batch
    for missing_batch in missing_batches:
        missing_batch_value = batch_to_numeric(missing_batch)
        
        # Calculate distance to each existing batch
        distances = {batch: abs(missing_batch_value - value) 
                    for batch, value in existing_batch_values.items()}
        
        # Find the 5 closest batches (or all available if fewer than 5)
        num_closest = min(5, len(distances))
        closest_batches = sorted(distances.items(), key=lambda x: x[1])[:num_closest]
        closest_batch_ids = [batch for batch, _ in closest_batches]
        
        # Calculate the mean of coefficients from the closest batches
        closest_coefficients = result_df.loc[closest_batch_ids, 'normalization_coefficient']
        imputed_coefficient = closest_coefficients.mean()
        
        # Add imputed coefficient to the result DataFrame
        result_df.loc[missing_batch, 'normalization_coefficient'] = imputed_coefficient
        
        # Add to list of imputed batches
        imputed_batches.append(missing_batch)
    
    return result_df, imputed_batches



def apply_normalization(data: pd.DataFrame, normalization_results: Dict[str, any]) -> pd.DataFrame:
    """
    Normalize the data using the calculated normalization coefficients.
    
    For each sample, this function:
    1. Extracts the batch number from the sample index
    2. Retrieves the corresponding normalization coefficient
    3. Divides the sample's features by this coefficient
    
    Parameters:
    -----------
    data: Original input DataFrame
    normalization_results: Results dictionary from normalize_batched_data function
    
    Returns:
    --------
    DataFrame with normalized data
    """
    # Create a copy of the input data to avoid modifying the original
    normalized_data = data.copy()
    
    # Get the normalization coefficients dataframe
    normalization_df = normalization_results['normalization_coefficients']
    
    # Process each sample
    for sample_idx in normalized_data.index:
        try:
            # Extract batch number from sample index
            batch = extract_batch_from_index(sample_idx)
            
            # Get normalization coefficient for this batch
            if batch in normalization_df.index:
                coefficient = normalization_df.loc[batch, 'normalization_coefficient']
                
                # Normalize the sample's features by dividing by the coefficient
                normalized_data.loc[sample_idx] = normalized_data.loc[sample_idx] / coefficient
            else:
                print(f"Warning: No normalization coefficient found for batch {batch}")
        
        except ValueError:
            print(f"Warning: Could not extract batch number from index: {sample_idx}")
    normalized_data = normalized_data.sort_index()

    return normalized_data



def plot_coefficient_comparison_methods(comparison_df, method1_name, method2_name, out_dir, y_padding=0.3):
    """
    Generate a detailed plot comparing normalization coefficients between methods.
    
    Parameters:
    -----------
    comparison_df: DataFrame containing coefficient comparison data
    method1_name: Name of first method for labeling
    method2_name: Name of second method for labeling
    y_padding: Factor to increase y-axis limits (default: 0.3 or 30% extra space)
    """

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar chart of coefficients by batch
    comparison_df[[method1_name, method2_name]].plot(
        kind='bar', 
        ax=ax1,
        color=[palette[0], palette[1]]
    )
    ax1.set_title('Normalization Coefficients by Batch')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_xlabel('Batch')
    ax1.legend([method1_name, method2_name])
    
    # Increase y-axis limits to make room for labels
    y_max = comparison_df[[method1_name, method2_name]].values.max()
    ax1.set_ylim(0, y_max * (1 + y_padding))
    
    # Plot 2: Ratio of coefficients
    comparison_df['ratio'].plot(kind='bar', ax=ax2, color=palette[4])
    ax2.set_title(f'Ratio of Coefficients ({method1_name} / {method2_name})')
    ax2.set_ylabel('Ratio')
    ax2.set_xlabel('Batch')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    
    # Add ratio values on top of bars
    for container in ax2.containers:
        labels = []
        for v in container:
            height = v.get_height()
            if height > 1.1 or height < 0.9:
                labels.append(f'{height:.2f}')
            else:
                labels.append('')  # Empty string for values between 0.9 and 1.1
        ax2.bar_label(container, labels=labels)
    
    # Increase y-axis limits to make room for labels
    ratio_max = comparison_df['ratio'].max()
    ax2.set_ylim(0, ratio_max * (1 + y_padding))
    
    # Add a text box with summary statistics
    # First, create a dummy legend to get its style properties
    legend = ax2.legend()
    legend_props = legend.get_frame()
    legend.remove()  # Remove the dummy legend

    # Now apply those properties to your text box
    stats_text = (
        f"Mean Ratio: {comparison_df['ratio'].mean():.2f}\n"
        f"Min Ratio: {comparison_df['ratio'].min():.2f}\n"
        f"Max Ratio: {comparison_df['ratio'].max():.2f}\n"
        f"CV: {comparison_df['ratio'].std() / comparison_df['ratio'].mean():.2f}"
    )

    # Create text box with legend-like properties
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(
                boxstyle='round',  # Legends typically have rounded corners
                facecolor=legend_props.get_facecolor(),
                edgecolor=legend_props.get_edgecolor(),
                alpha=legend_props.get_alpha(),
                linewidth=legend_props.get_linewidth(),
            )
            )
    
    plt.tight_layout()
    plt.savefig(out_dir + '/qc_vs_sample_normalization_ratios.png')



def batch_normalization(data, out_dir, nan_threshold, normalization_method, check_stability=False):
    """
    Perform batch-wise normalization of data using either QC samples or all samples as reference.
    
    This function compares different normalization methods, evaluates their stability,
    and returns the normalized dataset according to the specified method. It generates
    comparison plots and analysis results in the specified output directory.
    
    Parameters:
    ----------
    data : DataFrame
        Input data to be normalized. Expected to contain batch information and measurements.
    out_dir : str
        Output directory path where comparison plots and analysis results will be saved.
    nan_threshold : float
        Threshold for handling missing values during normalization. Features with NaN 
        values exceeding this threshold may be excluded.
    normalization_method : str
        Method to use for normalization. Valid values are 'QC' (using only quality 
        control samples) or 'Samples' (using all samples).
    check_stability : bool, optional
        Whether to perform stability checks during normalization. Default is False.
    
    Returns:
    -------
    DataFrame
        The normalized data according to the specified normalization method.
    
    Notes
    -----
    - QC Normalization: Uses only quality control samples to calculate normalization 
      coefficients, which are then applied to the entire dataset.
    - Sample Normalization: Uses all samples to calculate normalization coefficients.
    - The function evaluates the stability of normalization coefficients and provides 
      visual representations to help assess which method is more reliable.
    - Ensure the input data contains appropriate batch identifiers and QC sample indicators.
    - Invalid normalization method specifications will result in an error message.
    """

    # Compare stability options
    stability_dic = compare_stability_options(data, out_dir, nan_threshold, similarity_threshold=10)
    coeff_comparison = stability_dic['coefficient_comparison']
    plot_coefficient_comparison(coeff_comparison, out_dir)

    ## QC normalization
    # Calculate normalization coefficients based on QC samples
    results = normalize_batched_data_enhanced(
        data, nan_threshold, check_stability=check_stability)
    norm_coeffs1 = results['normalization_coefficients']
    # Apply normalization to get the normalized data
    qc_normalized_df = normalize_data_with_coefficients(data, norm_coeffs1)

    ## Sample Normalization
    # Calculate normalization coefficients based on all samples
    results = normalize_with_all_samples(
        data, nan_threshold, check_stability)
    norm_coeffs2 = results['normalization_coefficients']
    # Apply normalization to get the normalized data
    sample_normalized_df = normalize_data_with_coefficients(data, norm_coeffs2)

    # Compare normalization methods
    result = compare_normalization_methods(data, out_dir, qc_normalized_df, sample_normalized_df, 
                                    norm_coeffs1, norm_coeffs2,
                                    method1_name="QC Normalization", 
                                    method2_name="Sample Normalization",
                                    plot_results=True)
    
    # Return normalized data
    if normalization_method == 'QC':
        return qc_normalized_df
    elif normalization_method == 'Samples':
        return sample_normalized_df
    else:
        print('Error: Invalid Normalization Method')





