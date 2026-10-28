# Imports

# 1. Standard library
import gc
import os
import re

# 2. Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from typing import Dict, List, Optional, Tuple, Union

# Set seaborn theme and palette
sns.set_theme(style='white')
palette = sns.color_palette('deep')

# Set matplotlib backend to avoid crashes while plotting
matplotlib.use('Agg')

# Set global font sizes
plt.rcParams.update({
    'font.size': 18,          # Default text size
    'axes.titlesize': 20,     # Title size
    'axes.labelsize': 20,     # Axis label size
    'xtick.labelsize': 16,    # X-axis tick label size
    'ytick.labelsize': 16,    # Y-axis tick label size
    'legend.fontsize': 16,    # Legend text size
    'figure.titlesize': 22,    # Figure title size
    'axes.titleweight': 'bold', # Make titles bold
    'figure.titleweight': 'bold'
})



# Functions

def fit_regular_gmm(data: np.ndarray, n_components: int, 
                    random_state: int = 42) -> Tuple[GaussianMixture, float]:
    """
    Fit a regular Gaussian Mixture Model to the data.
    
    Parameters:
    -----------
    data : np.ndarray
        1D array of data points
    n_components : int
        Number of components to fit
    random_state : int, optional
        Random seed for reproducibility, by default 42
    
    Returns:
    --------
    Tuple[GaussianMixture, float]
        Fitted model and BIC score
    """
    # Reshape data to 2D for sklearn
    data_2d = data.reshape(-1, 1)
    
    # Initialize and fit the model
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(data_2d)
    
    return gmm, gmm.bic(data_2d)

def fit_bayesian_gmm(data: np.ndarray, n_components: int, 
                     weight_concentration_prior: float,
                     random_state: int = 42) -> Tuple[BayesianGaussianMixture, float]:
    """
    Fit a Bayesian Gaussian Mixture Model to the data.
    
    Parameters:
    -----------
    data : np.ndarray
        1D array of data points
    n_components : int
        Maximum number of components to fit
    weight_concentration_prior : float, optional
        The dirichlet concentration parameter, by default 0.01.
        Higher values favor more uniform weights, lower values allow more discrete components.
    random_state : int, optional
        Random seed for reproducibility, by default 42
    
    Returns:
    --------
    Tuple[BayesianGaussianMixture, float]
        Fitted model and BIC score
    """
    # Reshape data to 2D for sklearn
    data_2d = data.reshape(-1, 1)
    
    # Initialize and fit the model
    bgmm = BayesianGaussianMixture(
        n_components=n_components,
        weight_concentration_prior_type='dirichlet_distribution',
        weight_concentration_prior=weight_concentration_prior,
        random_state=random_state
    )
    bgmm.fit(data_2d)
    
    # Manual BIC calculation - this is the ONLY way to get BIC for Bayesian GMM
    n_samples = data_2d.shape[0]
    effective_components = np.sum(bgmm.weights_ > 0.01)
    n_params = effective_components * 3 - 1
    log_likelihood = np.sum(bgmm.score_samples(data_2d))
    bic = -2 * log_likelihood + n_params * np.log(n_samples)

    return bgmm, bic

def find_optimal_components(data: np.ndarray, max_components: int = 8, 
                           model_type: str = 'regular', 
                           weight_concentration_prior: float = 0.01,
                           random_state: int = 42) -> Tuple[Union[GaussianMixture, BayesianGaussianMixture], int, float]:
    """
    Find the optimal number of components for GMM using BIC.
    
    Parameters:
    -----------
    data : np.ndarray
        1D array of data points
    max_components : int, optional
        Maximum number of components to try, by default 8
    weight_concentration_prior : float, optional
        Dirichlet concentration parameter passed to the Bayesian GMM, by default 0.1
    model_type : str, optional
        Type of GMM to use ('regular' or 'bayesian'), by default 'regular'
    random_state : int, optional
        Random seed for reproducibility, by default 42
    
    Returns:
    --------
    Tuple[Union[GaussianMixture, BayesianGaussianMixture], int, float]
        Best model, optimal number of components, and best BIC score
    """
    best_bic = np.inf
    best_model = None
    best_n_components = 1
    
    
    # Try different numbers of components from 1 to max_components
    for n_components in range(1, max_components + 1):
        if model_type == 'regular':
            model, bic = fit_regular_gmm(data, n_components, random_state)
        else:  # bayesian
            model, bic = fit_bayesian_gmm(data, n_components,weight_concentration_prior, random_state=42)
        
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_n_components = n_components
    
    return best_model, best_n_components, best_bic

def classify_components(means: np.ndarray, weights: np.ndarray, 
                       weight_noise_threshold: float = 0.2) -> List[str]:
    """
    Classify GMM components based on means and weights.
    
    Parameters:
    -----------
    means : np.ndarray
        Array of component means
    weights : np.ndarray
        Array of component weights
    weight_noise_threshold : float, optional
        Threshold for classifying a component as noise, by default 0.2
    
    Returns:
    --------
    List[str]
        List of classifications for each component
    """
    n_components = len(means)
    classifications = [''] * n_components
    
    # Sort components by mean
    sorted_indices = np.argsort(means)
    
    # Apply classification rules based on number of components
    if n_components == 1:
        # Single component case: always Full Peak Integration
        classifications[sorted_indices[0]] = 'Full Peak Integration'
    
    elif n_components == 2:
        # Two component case
        # Highest weight component: Full Peak Integration
        highest_weight_idx = np.argmax(weights)
        classifications[highest_weight_idx] = 'Full Peak Integration'
        
        # Other component
        other_idx = 1 - highest_weight_idx  # If highest_weight_idx is 0, other is 1; if highest_weight_idx is 1, other is 0
        
        # If full peak integration has the highest mean
        if means[highest_weight_idx] >= means[other_idx]:
            # If other component weight is below threshold, classify as Noise
            if weights[other_idx] < weight_noise_threshold:
                classifications[other_idx] = 'Noise'
            else:
                classifications[other_idx] = 'Partial Peak Integration'
        else:
            # If full peak integration does not have the highest mean
            classifications[other_idx] = 'Multiple Peak Integration'
    
    else:
        # Three or more component case
        # Lowest mean component: Noise or Partial Peak Integration based on weight
        lowest_mean_idx = sorted_indices[0]
        if weights[lowest_mean_idx] < weight_noise_threshold:
            classifications[lowest_mean_idx] = 'Noise'
            noise_mean = means[lowest_mean_idx]
        else:
            classifications[lowest_mean_idx] = 'Partial Peak Integration'
            noise_mean = -np.inf  # No noise component
        
        # Find highest weight component: Full Peak Integration
        highest_weight_idx = np.argmax(weights)
        classifications[highest_weight_idx] = 'Full Peak Integration'
        full_peak_mean = means[highest_weight_idx]
        
        # Process remaining components
        for i in range(n_components):
            # Skip components already classified
            if classifications[i]:
                continue
                
            if means[i] > noise_mean and means[i] < full_peak_mean:
                classifications[i] = 'Partial Peak Integration'
            elif means[i] > full_peak_mean:
                classifications[i] = 'Multiple Peak Integration'
    
    return classifications

def analyze_feature(data: np.ndarray, feature_name: str, max_components: int = 10, 
                   model_type: str = 'both', weight_noise_threshold: float = 0.2,
                   weight_concentration_prior: float = 0.01,
                   random_state: int = 42) -> Dict:
    """
    Analyze a feature using GMM and classify components.
    
    Parameters:
    -----------
    data : np.ndarray
        1D array of data points for a specific feature
    feature_name : str
        Name of the feature being analyzed
    max_components : int, optional
        Maximum number of components to try, by default 10
    model_type : str, optional
        Type of GMM to use ('regular', 'bayesian', or 'both'), by default 'both'
    weight_noise_threshold : float, optional
        Threshold for classifying a component as noise, by default 0.2
    weight_concentration_prior : float, optional
        Dirichlet concentration parameter for the Bayesian GMM, by default 0.01
    random_state : int, optional
        Random seed for reproducibility, by default 42
    
    Returns:
    --------
    Dict
        Dictionary containing analysis results
    """
    # Clean data (remove NaN values)
    clean_data = data[~np.isnan(data)]
    
    results = {'feature_name': feature_name}
    
    # Fit models based on model_type
    if model_type in ['regular', 'both']:
        reg_model, reg_n_components, reg_bic = find_optimal_components(
            clean_data, max_components, 'regular', random_state)
        
        # Extract parameters from the regular model
        reg_means = reg_model.means_.flatten()
        reg_weights = reg_model.weights_
        reg_stds = np.sqrt(reg_model.covariances_.flatten())
        
        # Classify regular GMM components
        reg_classifications = classify_components(reg_means, reg_weights, weight_noise_threshold)
        
        # Store regular GMM results
        results['regular'] = {
            'model': reg_model,
            'n_components': reg_n_components,
            'bic': reg_bic,
            'means': reg_means,
            'weights': reg_weights,
            'stds': reg_stds,
            'classifications': reg_classifications
        }
    
    if model_type in ['bayesian', 'both']:
        bay_model, bay_n_components, bay_bic = find_optimal_components(
            clean_data, max_components, 'bayesian', weight_concentration_prior, random_state)
        
        # Extract parameters from the Bayesian model
        bay_means = bay_model.means_.flatten()
        bay_weights = bay_model.weights_
        bay_stds = np.sqrt(bay_model.covariances_.flatten())
        
        # Classify Bayesian GMM components
        bay_classifications = classify_components(bay_means, bay_weights, weight_noise_threshold)
        
        # Store Bayesian GMM results
        results['bayesian'] = {
            'model': bay_model,
            'n_components': bay_n_components,
            'bic': bay_bic,
            'means': bay_means,
            'weights': bay_weights,
            'stds': bay_stds,
            'classifications': bay_classifications
        }
    
    results['data'] = clean_data
    
    return results

def create_summary_dataframe(results_dict: Dict) -> pd.DataFrame:
    """
    Create a summary dataframe from analysis results.
    
    Parameters:
    -----------
    results_dict : Dict
        Dictionary of analysis results for multiple features
    
    Returns:
    --------
    pd.DataFrame
        Summary dataframe with key information
    """
    summary_data = []
    
    for feature_name, result in results_dict.items():
        for model_type in ['regular', 'bayesian']:
            if model_type in result:
                model_result = result[model_type]
                
                # Get counts of each classification type
                classifications = model_result['classifications']
                n_noise = classifications.count('Noise')
                n_partial = classifications.count('Partial Peak Integration')
                n_full = classifications.count('Full Peak Integration')
                n_multiple = classifications.count('Multiple Peak Integration')
                
                # Create a row for each component
                for i in range(model_result['n_components']):
                    component_data = {
                        'Feature': feature_name,
                        'Model Type': model_type,
                        'BIC': model_result['bic'],
                        'Total Components': model_result['n_components'],
                        'Component Index': i,
                        'Mean': model_result['means'][i],
                        'Weight': model_result['weights'][i],
                        'Std Dev': model_result['stds'][i],
                        'Classification': model_result['classifications'][i],
                        'Noise Components': n_noise,
                        'Partial Peak Components': n_partial,
                        'Full Peak Components': n_full,
                        'Multiple Peak Components': n_multiple
                    }
                    summary_data.append(component_data)
    
    return pd.DataFrame(summary_data)

def plot_gmm_fit(results: Dict, feature_name: str, out_dir: str):
    """
    Plot GMM model fits for a feature.
    
    Parameters:
    -----------
    results : Dict
        Analysis results for a feature
    feature_name : str
        Name of the feature
    out_dir : str
        Output directory for plots
    """

    data = results['data']
    model_types = [m for m in ['regular', 'bayesian'] if m in results]
    title_name = feature_name.split('_')[1]
    # Define classification color mapping with requested colors
    class_colors = {
        'Noise': 'red',
        'Partial Peak Integration': 'orange',
        'Full Peak Integration': 'green',
        'Multiple Peak Integration': 'purple'
    }
    
    # Determine if we need one plot or two side-by-side
    if len(model_types) > 1:  # Both regular and Bayesian
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'GMM Fit for {title_name}', fontsize=16)
    else:  # Just one model type
        fig, axes = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
        axes = [axes]  # Make it a list for consistent indexing
        fig.suptitle(f'{model_types[0].capitalize()} GMM Fit for {title_name}', y=0.95)    

    # Adaptive resolution - use fewer points for wider ranges
    data_range = max(data) - min(data)
    n_points = min(1000, max(100, int(10000 / data_range)))

    # Plot for each model type
    for i, model_type in enumerate(model_types):
        ax = axes[i]
        
        sns.histplot(data, bins=100, kde=False, stat='density', alpha=0.3, ax=ax)
        ax.grid(False)  # Add this line to explicitly remove the grid        
        # Calculate where Gaussian curves reach ~zero (mean ± 4*std for all components)
        model = results[model_type]['model']
        means = model.means_.flatten()
        stds = np.sqrt(model.covariances_.flatten())

        curve_min = np.min(means - 4 * stds)
        curve_max = np.max(means + 4 * stds)

        # Extend xlim to show full curves
        current_xlim = ax.get_xlim()
        extended_min = min(current_xlim[0], curve_min)
        extended_max = max(current_xlim[1], curve_max)
        ax.set_xlim(extended_min, extended_max)

        # Create x range for plotting
        x = np.linspace(extended_min, extended_max, n_points)

        model_result = results[model_type]
        model = model_result['model']
        classifications = model_result['classifications']
        weights = model_result['weights']
        means = model_result['means']
        stds = model_result['stds']
        
        # Get overall density
        densities = np.zeros_like(x)
        component_densities = np.zeros((len(x), model.n_components))
        

        for j in range(model.n_components):
            # Calculate density for this component
            mean = means[j]
            std = stds[j]
            weight = weights[j]
            classification = classifications[j]
            if weight < 0.01:
                continue
            
            component_density = weight * norm.pdf(x, mean, std)
            component_densities[:, j] = component_density
            densities += component_density

            # Shade area under component density with classification color
            ax.fill_between(x, component_density, alpha=0.3,
                           color=class_colors[classification],
                           label=f"{classification}: μ={mean:.2f}, w={weight:.2f}, σ={std:.2f}")
            # Plot component density line
            ax.plot(x, component_density, color=class_colors[classification], alpha=0.7)
        
        # Plot overall density
        ax.plot(x, densities, color='black', linewidth=2, 
               label=f"Overall Density", alpha=0.7)
        # Add title, labels to each subplot
        ax.set_xlabel('log2(Signal Intensity)')
        ax.set_ylabel('Density')
        
        # Sort legend by decreasing weight
        handles, labels = ax.get_legend_handles_labels()
        weights_for_sorting = []
        for label in labels:
            if 'w=' in label:
                weight = float(label.split('w=')[1].split(',')[0])
                weights_for_sorting.append(weight)
            else:
                weights_for_sorting.append(0)  # For overall density label
        
        sorted_indices = np.argsort(weights_for_sorting)[::-1]
        sorted_handles = [handles[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        
        # Find the best legend location based on the histogram data
        # By checking where the data density is lowest
        hist, bin_edges = np.histogram(data, bins=20, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate average density in each quadrant of the plot
        x_mid = (min(data) + max(data)) / 2
        y_max = max(hist) * 1.1  # Slightly higher than the max histogram height
        
        # Get indices for left and right halves
        left_indices = bin_centers <= x_mid
        right_indices = bin_centers > x_mid

  
        # Calculate density in each quadrant (upper/lower left/right)
        if len(hist[left_indices]) > 0:
            upper_left_density = np.mean(hist[left_indices]) if np.any(hist[left_indices] > y_max/2) else 0
            lower_left_density = np.mean(hist[left_indices]) if np.any(hist[left_indices] <= y_max/2) else 0
        else:
            upper_left_density = lower_left_density = float('inf')  # Avoid empty regions
            
        if len(hist[right_indices]) > 0:
            upper_right_density = np.mean(hist[right_indices]) if np.any(hist[right_indices] > y_max/2) else 0
            lower_right_density = np.mean(hist[right_indices]) if np.any(hist[right_indices] <= y_max/2) else 0
        else:
            upper_right_density = lower_right_density = float('inf')  # Avoid empty regions
        
        # Find quadrant with lowest density for legend placement
        densities_by_quadrant = {
            'upper left': upper_left_density,
            'upper right': upper_right_density,
            'lower left': lower_left_density,
            'lower right': lower_right_density
        }
        
        # Get best location (with lowest density)
        best_location = min(densities_by_quadrant, key=densities_by_quadrant.get)

        # Create legend with proper location
        ax.legend(sorted_handles, sorted_labels, loc=best_location, framealpha=0.8)
    

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, feature_name) + '.png')
    plt.clf()
    plt.close('all')
    gc.collect()

def analyze_tabular_data(data: pd.DataFrame, model_type: str = 'both', 
                       weight_noise_threshold: float = 0.2, max_components: int = 6,
                       features: List[str] = None, weight_concentration_prior: float = 0.01,
                       random_state: int = 42) -> Tuple[Dict, pd.DataFrame]:
    """
    Analyze multiple features in tabular data using GMM.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Tabular data with features as columns
    features : List[str], optional
        List of feature names to analyze. If None, all columns will be used.
    max_components : int, optional
        Maximum number of components to try, by default 10
    model_type : str, optional
        Type of GMM to use ('regular', 'bayesian', or 'both'), by default 'both'
    weight_noise_threshold : float, optional
        Threshold for classifying a component as noise, by default 0.2
    weight_concentration_prior : float, optional
        The dirichlet concentration parameter, by default 0.01.
        Higher values favor more uniform weights, lower values allow more discrete components.
    random_state : int, optional
        Random seed for reproducibility, by default 42
    
    Returns:
    --------
    Tuple[Dict, pd.DataFrame]
        Dictionary of analysis results and summary dataframe
    """
    # If no features specified, use all columns
    if features is None:
        features = data.columns.tolist()
    
    # Analyze each feature
    results_dict = {}
    for feature in features:
        feature_data = data[feature].values
        if np.isnan(feature_data).all():
            continue
        else:
            result = analyze_feature(
                feature_data, feature, max_components, model_type, 
                weight_noise_threshold, weight_concentration_prior, random_state
            )
            results_dict[feature] = result
    
    # Create summary dataframe
    summary_df = create_summary_dataframe(results_dict)
    
    return results_dict, summary_df

def evaluate_results(results_dict: Dict, sub_dir: str, plot: bool):
    """
    Evaluate and visualize GMM results for all features.

    Parameters:
    -----------
    results_dict : Dict
        Dictionary of analysis results for multiple features
    sub_dir : str
        Output directory for plots
    plot : bool
        Whether to generate GMM fit plots for each feature
    """
    if plot:
        # Plot GMM fits for each feature
        for feature_name, result in results_dict.items():
            plot_gmm_fit(result, feature_name, sub_dir)
    else:
        print('No GMM plots generated')


def get_component_probabilities(results_dict: Dict, data: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """
    Extracts component membership probabilities for each datapoint across all features.
    
    For each feature:
    1. Uses the specified model type (regular, bayesian, or both)
    2. Gets data, drops NaN values but preserves original indices
    3. Calculates component membership probabilities
    
    Returns a DataFrame with samples as rows and component probabilities as columns,
    aggregated across all features.
    
    Parameters:
    -----------
    results_dict : Dict
        Dictionary of analysis results for multiple features, as produced by analyze_tabular_data
    data : pd.DataFrame
        The original data containing all features
    model_type : str, optional
        Type of GMM to use ('regular', 'bayesian', or 'both'), by default 'both'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with component membership probabilities for each datapoint
    """
    # Validate model_type
    if model_type not in ['regular', 'bayesian', 'both']:
        raise ValueError("model_type must be 'regular', 'bayesian', or 'both'")
        
    # Initialize an empty list to store dataframes for each feature
    feature_dfs = []
    
    # Process each feature
    for feature_name, result in results_dict.items():
        # Skip if feature not in data
        if feature_name not in data.columns:
            continue
        
        # Determine which model types to use based on user preference and availability
        available_model_types = [m for m in ['regular', 'bayesian'] if m in result]
        if not available_model_types:
            print(f"Warning: No models available for feature '{feature_name}'. Skipping.")
            continue
            
        if model_type == 'both':
            # Use all available model types
            selected_model_types = available_model_types
        else:
            # Use only the specified model type if available
            if model_type in available_model_types:
                selected_model_types = [model_type]
            else:
                print(f"Warning: {model_type} model not available for feature '{feature_name}'. Skipping.")
                continue
            
        # Get feature data
        feature_data = data[feature_name]
        
        # Identify non-NaN values and their indices
        non_nan_mask = ~feature_data.isna()
        valid_indices = feature_data.index[non_nan_mask]
        
        # Skip if no valid data
        if len(valid_indices) == 0:
            print(f"Warning: No valid data for feature '{feature_name}'. Skipping.")
            continue
        
        # Get clean data for processing
        clean_data = feature_data[non_nan_mask].values.reshape(-1, 1)
        
        # Initialize dataframe for this feature with the correct index
        feature_df = pd.DataFrame(index=valid_indices)
        
        # Process each selected model type
        for current_model_type in selected_model_types:
            model_result = result[current_model_type]
            model = model_result['model']
            classifications = model_result['classifications']
            
            # Calculate component probabilities
            probabilities = model.predict_proba(clean_data)
            
            # Add columns for each component's probability
            for i in range(model_result['n_components']):
                # Create clear column name with feature, model type, component index, and classification
                classification_abbr = ''.join([word[0] for word in classifications[i].split()])
                col_name = f"{feature_name}_{current_model_type}_comp{i}_{classification_abbr}_prob"
                
                # Add probability column to feature dataframe
                feature_df[col_name] = probabilities[:, i]
            
            # Add most likely component and its probability
            most_likely_comp = np.argmax(probabilities, axis=1)
            feature_df[f"{feature_name}_{current_model_type}_most_likely_comp"] = most_likely_comp
            feature_df[f"{feature_name}_{current_model_type}_max_prob"] = np.max(probabilities, axis=1)
            
            # Add classification labels
            most_likely_classifications = [classifications[idx] for idx in most_likely_comp]
            feature_df[f"{feature_name}_{current_model_type}_classification"] = most_likely_classifications
        
        # Add this feature's dataframe to our list
        feature_dfs.append(feature_df)
    
    # If no valid features, return empty dataframe
    if not feature_dfs:
        return pd.DataFrame()
    
    # Combine all feature dataframes
    # Using outer join to keep all sample indices from all features
    combined_df = pd.concat(feature_dfs, axis=1)
    
    return combined_df

def filter_noise_components(original_data: pd.DataFrame, 
                           probability_df: pd.DataFrame, 
                           noise_proba_threshold: float = 0.8) -> pd.DataFrame:
    """
    Identifies datapoints classified as noise with high probability and sets them to NaN in the original data.
    
    The function examines the probability dataframe (from get_component_probabilities) and looks for 
    datapoints where the probability of belonging to a Noise component exceeds the threshold.
    These points are then set to NaN in the original data.
    
    Parameters:
    -----------
    original_data : pd.DataFrame
        The original data that will be modified
    probability_df : pd.DataFrame
        DataFrame containing component membership probabilities (from get_component_probabilities)
    noise_proba_threshold : float, optional
        Probability threshold above which to classify a point as noise, by default 0.8
        
    Returns:
    --------
    pd.DataFrame
        A copy of the original data with high-probability noise points set to NaN
    """
    # Create a copy of the original data to avoid modifying the input
    filtered_data = original_data.copy()
    
    # Get a list of all the features in the original data
    features = original_data.columns
    
    # Process each feature
    for feature in features:
        # Find the relevant columns in the probability dataframe for this feature
        # We're looking for columns that contain:
        # 1. The feature name
        # 2. "N_prob" (abbreviated from "Noise_prob" in the probability dataframe)
        noise_prob_cols = [col for col in probability_df.columns 
                          if feature in col and 'N_prob' in col]
        
        # Skip if no noise probability columns were found for this feature
        if not noise_prob_cols:
            continue
        
        # Get the indices of datapoints to be filtered (set to NaN)
        indices_to_filter = set()
        
        for col in noise_prob_cols:
            # Find indices where the noise probability exceeds the threshold
            high_noise_indices = probability_df[probability_df[col] > noise_proba_threshold].index
            indices_to_filter.update(high_noise_indices)
        
        # We can also look at the classification columns directly
        classification_cols = [col for col in probability_df.columns 
                              if feature in col and 'classification' in col]
        
        for col in classification_cols:
            # Find indices where the classification is "Noise" with high probability
            # First, get the corresponding probability column
            base_col_name = col.replace('_classification', '')
            max_prob_col = f"{base_col_name}_max_prob"
            
            if max_prob_col in probability_df.columns:
                # Get indices where classification is "Noise" and probability is high
                noise_indices = probability_df[
                    (probability_df[col] == 'Noise') & 
                    (probability_df[max_prob_col] > noise_proba_threshold)
                ].index
                
                indices_to_filter.update(noise_indices)
        
        # Set the identified indices to NaN in the filtered data
        if indices_to_filter:
            # Make sure the indices exist in the filtered data (handle potential mismatches)
            valid_indices = [idx for idx in indices_to_filter if idx in filtered_data.index]
            
            if valid_indices:
                filtered_data.loc[valid_indices, feature] = np.nan
    
    return filtered_data

def get_filtered_points(data: pd.DataFrame, filtered_data: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies datapoints that were changed to NaN during the filtering process.
    
    This function compares the original data with the filtered data and finds all points
    that were converted to NaN. It returns a DataFrame containing the indices and column names
    of these changed points.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The original data before filtering
    filtered_data : pd.DataFrame
        The filtered data after noise removal
        
    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns 'index' and 'feature' that identifies each changed datapoint
    """
    # Ensure both dataframes have the same structure
    if data.shape != filtered_data.shape:
        raise ValueError("Original and filtered data must have the same shape")
        
    if not all(data.columns == filtered_data.columns):
        raise ValueError("Original and filtered data must have the same columns")
        
    if not all(data.index == filtered_data.index):
        raise ValueError("Original and filtered data must have the same indices")
    
    # List to store information about changed points
    changed_points = []
    
    # Check each column
    for column in data.columns:
        # Find points that were not NaN in original but are NaN in filtered
        changed_mask = ~data[column].isna() & filtered_data[column].isna()
        
        # Get the indices where changes occurred
        changed_indices = data.index[changed_mask]
        
        # Store information about each changed point
        for idx in changed_indices:
            changed_points.append({
                'index': idx,
                'feature': column,
                'original_value': data.loc[idx, column]
            })
    
    # Convert the list to a DataFrame
    if changed_points:
        result_df = pd.DataFrame(changed_points)
        
        return result_df
    else:
        return pd.DataFrame(columns=['index', 'feature', 'original_value'])



def calculate_gmm_pvalues(results_dict: Dict, new_data: pd.DataFrame, 
                         model_type: str = 'bayesian',
                         min_std: float = 1e-10) -> pd.DataFrame:
    """
    Calculates two-sided p-values for new samples based on the fitted GMM.
    
    Uses the analytical CDF of the GMM (weighted sum of component Gaussian CDFs)
    to efficiently calculate p-values without sampling.
    
    Parameters:
    -----------
    results_dict : Dict
        Dictionary of analysis results for multiple features, as produced by analyze_tabular_data
    new_data : pd.DataFrame
        DataFrame containing new samples to evaluate
    model_type : str, optional
        Type of GMM to use ('regular', 'bayesian'), by default 'bayesian'
    min_std : float, optional
        Minimum standard deviation to prevent numerical issues, by default 1e-10
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with p-values for each sample and feature
    """
    # Check if model_type is valid
    if model_type not in ['regular', 'bayesian']:
        raise ValueError("model_type must be 'regular' or 'bayesian'")
    
    # Initialize a DataFrame to store p-values with the same index as new_data
    p_values_df = pd.DataFrame(index=new_data.index)
    
    # Process each feature
    for feature_name, result in results_dict.items():
        # Skip if feature is not in new_data
        if feature_name not in new_data.columns:
            continue
            
        # Skip if model type is not available
        if model_type not in result:
            print(f"Warning: {model_type} model not available for feature '{feature_name}'. Skipping.")
            continue
        
        try:
            # Get the model and extract its parameters
            model_result = result[model_type]
            model = model_result['model']
            means = model.means_.flatten()
            
            # Handle covariances carefully to avoid numerical issues
            stds = np.sqrt(model.covariances_.flatten())
            # Ensure all standard deviations are at least min_std
            stds = np.maximum(stds, min_std)
            
            weights = model.weights_
            # Ensure weights sum to 1 (they should, but let's be safe)
            if not np.isclose(np.sum(weights), 1.0):
                weights = weights / np.sum(weights)
            
            # Get new data for this feature, handling NaNs
            feature_data = new_data[feature_name].values
            non_nan_mask = ~np.isnan(feature_data)
            valid_indices = np.where(non_nan_mask)[0]
            valid_values = feature_data[non_nan_mask]
            
            # Skip if no valid data
            if len(valid_values) == 0:
                p_values_df[f"{feature_name}_pvalue"] = np.nan
                continue
            
            # Initialize arrays for p-values and CDF positions
            p_values = np.full(len(feature_data), np.nan)
            cdf_positions = np.full(len(feature_data), np.nan)
            
            # Calculate p-values for each valid value
            for i, value in enumerate(valid_values):
                # Calculate CDF position using a more robust approach
                # Use norm.cdf instead of manual erf calculation
                component_cdfs = np.array([norm.cdf(value, loc=mean, scale=std) 
                                          for mean, std in zip(means, stds)])
                
                # Compute weighted sum for the mixture CDF
                cdf_position = np.dot(weights, component_cdfs)
                
                # Ensure CDF position is strictly within [0, 1] to avoid numerical issues
                cdf_position = np.clip(cdf_position, 0.0, 1.0)
                
                # Store CDF position
                idx = valid_indices[i]
                cdf_positions[idx] = cdf_position
                
                # Calculate two-sided p-value
                # For two-sided test, we take the minimum of the position and its complement, then multiply by 2
                p_value = 2.0 * min(cdf_position, 1.0 - cdf_position)
                
                # Ensure p-value is within [0, 1]
                p_value = min(p_value, 1.0)
                
                # Double-check that p-value is not negative
                if p_value < 0:
                    print(f"Warning: Negative p-value calculated for feature '{feature_name}', value {value}.")
                    print(f"CDF position: {cdf_position}, Component CDFs: {component_cdfs}")
                    p_value = 0.0  # Force to valid range
                
                # Store p-value
                p_values[idx] = p_value
            
            # Add p-values and CDF positions to result DataFrame
            p_values_df[f"{feature_name}_pvalue"] = p_values
            p_values_df[f"{feature_name}_cdf_position"] = cdf_positions
            
            # Also store the log density for reference
            log_density_values = np.full(len(feature_data), np.nan)
            for i, value in enumerate(valid_values):
                try:
                    value_reshaped = np.array([[value]])
                    log_density = model.score_samples(value_reshaped)[0]
                    idx = valid_indices[i]
                    log_density_values[idx] = log_density
                except Exception as e:
                    print(f"Warning: Error calculating log density for feature '{feature_name}', value {value}: {e}")
                    
            p_values_df[f"{feature_name}_log_density"] = log_density_values
            
        except Exception as e:
            print(f"Error processing feature '{feature_name}': {e}")
            continue
    
    return p_values_df


def calculate_gmm_pvalues_for_triplicates(results_dict: Dict, new_data: pd.DataFrame, 
                                         model_type: str = 'bayesian',
                                         min_std: float = 1e-10,
                                         suffix_pattern: str = r'_[123]$') -> pd.DataFrame:
    """
    Calculates two-sided p-values based on the mean of sample triplicates using a fitted GMM.
    
    Samples with suffixes (_1, _2, _3) are grouped by their base name (without suffix),
    averaged, and then p-values are calculated on these mean values.
    
    Parameters:
    -----------
    results_dict : Dict
        Dictionary of analysis results for multiple features, as produced by analyze_tabular_data
    new_data : pd.DataFrame
        DataFrame containing samples with triplicate measurements to evaluate
    model_type : str, optional
        Type of GMM to use ('regular', 'bayesian'), by default 'bayesian'
    min_std : float, optional
        Minimum standard deviation to prevent numerical issues, by default 1e-10
    suffix_pattern : str, optional
        Regular expression pattern to identify and remove suffixes from sample names, 
        by default r'_[123]$' (matches _1, _2, _3 at the end of the string)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with p-values for each mean sample and feature
    """
    
    # Function to extract base sample name (remove suffix)
    def get_base_sample_name(sample_name):
        return re.sub(suffix_pattern, '', str(sample_name))
    
    # Create a mapping from sample to base name
    sample_mapping = pd.Series(index=new_data.index, data=[get_base_sample_name(idx) for idx in new_data.index])
    
    # Group by base sample name and calculate mean for each feature
    grouped_data = new_data.groupby(sample_mapping).mean()
    
    # Now use the original function to calculate p-values on the grouped means
    return calculate_gmm_pvalues(results_dict, grouped_data, model_type, min_std)



def calculate_mode_probabilities(results_dict: Dict, p_values_df: pd.DataFrame, 
                            data: pd.DataFrame, significance_threshold: float = 0.05,
                            model_type: str = 'bayesian', n_samples: int = 10000,
                            random_state: int = 42) -> pd.DataFrame:
    """
    For significant p-values, calculates the probability that each point belongs to 
    the Noise mode and Multiple Peak Integration mode, enhanced with sampling-based context.
    
    Parameters:
    ------------
    results_dict : Dict
        Dictionary of analysis results for multiple features, as produced by analyze_tabular_data
    p_values_df : pd.DataFrame
        DataFrame containing p-values for each sample and feature, as produced by
        calculate_gmm_pvalues
    data : pd.DataFrame
        The original data containing all features
    significance_threshold : float, optional
        P-value threshold below which a point is considered significant, by default 0.05
    model_type : str, optional
        Type of GMM to use ('regular' or 'bayesian'), by default 'bayesian'
    n_samples : int, optional
        Number of samples drawn per component for density distribution estimation,
        by default 10000
    random_state : int, optional
        Random seed for sampling from the GMM components, by default 42

    Returns:
    ---------
    pd.DataFrame
    DataFrame containing, for each significant data point:
    - 'feature': feature name
    - 'value': observed value
    - 'p_value': p-value from GMM
    - 'adjusted_p_value': BH-adjusted p-value (if available)
    - 'noise_density': raw density contribution from Noise components
    - 'multi_peak_density': raw density contribution from Multiple Peak Integration components
    - 'noise_probability': normalized probability of belonging to Noise
    - 'multi_peak_probability': normalized probability of belonging to Multiple Peak Integration
    - 'noise_percentile': percentile of the point in the Noise component density distribution
    - 'noise_confidence': confidence level of Noise classification ('low', 'medium', 'high')
    - 'multi_peak_percentile': percentile of the point in the Multiple Peak Integration density distribution
    - 'multi_peak_confidence': confidence level of Multiple Peak Integration classification
    Sorted by ascending p-value.
    """
    rng = np.random.default_rng(random_state)

    # Initialize result DataFrame
    result_df = pd.DataFrame()
    
    # Get p-value columns
    p_value_cols = [col for col in p_values_df.columns if col.endswith('_pvalue')]
    
    # Track rows to include in the final result
    rows_to_include = []
    
    # Process each feature
    for p_value_col in p_value_cols:
        # Get feature name from p-value column
        feature_name = p_value_col.replace('_pvalue', '')
        

        # Skip if feature not in results_dict or data
        if feature_name not in results_dict or feature_name not in data.columns:
            continue
            
        # Skip if model type not available for this feature
        if model_type not in results_dict[feature_name]:
            continue
            
        # Get model result dictionary
        model_result = results_dict[feature_name][model_type]
        model = model_result['model']  # The actual model object
        classifications = model_result['classifications']  # Component classifications
        means = model_result['means']  # Component means
        stds = model_result['stds']    # Component standard deviations
        
        # Find indices of Noise and Multiple Peak Integration components
        noise_indices = [i for i, c in enumerate(classifications) if c == 'Noise']
        multi_peak_indices = [i for i, c in enumerate(classifications) if c == 'Multiple Peak Integration']
        
        # Find significant p-values for this feature
        significant_mask = p_values_df[p_value_col] < significance_threshold
        significant_indices = p_values_df.index[significant_mask]
        
        # Skip if no significant p-values
        if len(significant_indices) == 0:
            continue
            
        # Get values for significant points
        significant_values = data.loc[significant_indices, feature_name]
        
        # Remove NaN values
        valid_mask = ~significant_values.isna()
        valid_indices = significant_indices[valid_mask]
        valid_values = significant_values[valid_mask].values
        
        # Skip if no valid values
        if len(valid_values) == 0:
            continue
        
        # Generate samples from the GMM to establish density distributions
        component_samples = {}
        
        # Only proceed with sampling if there are components of each type
        if noise_indices:
            # Sample from Noise components using the means and stds arrays
            noise_samples = []
            for i in noise_indices:
                # Generate samples from normal distribution using mean and std
                samples = rng.normal(
                    means[i],            # Mean for this component
                    stds[i],             # Standard deviation for this component
                    n_samples // len(noise_indices)
                )
                noise_samples.append(samples)
            
            # Combine all noise samples
            noise_samples = np.concatenate(noise_samples)
            component_samples['noise'] = noise_samples
        
        if multi_peak_indices:
            # Sample from Multiple Peak Integration components
            multi_peak_samples = []
            for i in multi_peak_indices:
                # Generate samples from normal distribution using mean and std
                samples = rng.normal(
                    means[i],            # Mean for this component
                    stds[i],             # Standard deviation for this component
                    n_samples // len(multi_peak_indices)
                )
                multi_peak_samples.append(samples)
            
            # Combine all multi-peak samples
            multi_peak_samples = np.concatenate(multi_peak_samples)
            component_samples['multi_peak'] = multi_peak_samples
        
        # Calculate log density distributions for each component type
        log_density_distributions = {}
        
        for comp_type, samples in component_samples.items():
            # Reshape samples for density calculation with the model
            samples_reshaped = samples.reshape(-1, 1)
            
            # Get log probability densities for all samples
            log_densities = model._estimate_log_prob(samples_reshaped)
            
            if comp_type == 'noise':
                # Sum log densities from noise components
                comp_log_densities = np.array([
                    logsumexp(row[noise_indices]) for row in log_densities
                ])
                log_density_distributions['noise'] = comp_log_densities
                
            elif comp_type == 'multi_peak':
                # Sum log densities from multi peak components
                comp_log_densities = np.array([
                    logsumexp(row[multi_peak_indices]) for row in log_densities
                ])
                log_density_distributions['multi_peak'] = comp_log_densities
        
        # Create a temporary DataFrame for this feature
        temp_df = pd.DataFrame(index=valid_indices)
        temp_df['feature'] = feature_name
        temp_df['value'] = valid_values
        temp_df['p_value'] = p_values_df.loc[valid_indices, p_value_col].values
        
        # If adjusted p-values are available, include them
        adj_p_col = f"{feature_name}_adjusted_pvalue"
        if adj_p_col in p_values_df.columns:
            temp_df['adjusted_p_value'] = p_values_df.loc[valid_indices, adj_p_col].values
        
        # Calculate component probabilities for each valid value
        for i, value in enumerate(valid_values):
            # Reshape for log_prob calculation
            value_reshaped = np.array([[value]])
            
            # Get the raw log-density values for each component
            log_density = model._estimate_log_prob(value_reshaped)[0]
            # Convert to actual density values
            density = np.exp(log_density)
            
            # Sum the density values for each type of component
            noise_density = sum(density[j] for j in noise_indices) if noise_indices else 0
            multi_peak_density = sum(density[j] for j in multi_peak_indices) if multi_peak_indices else 0
            
            # Calculate the total density for normalization
            total_density = noise_density + multi_peak_density
            
            # Index for the current point
            idx = valid_indices[i]
            
            # Store raw densities
            temp_df.loc[idx, 'noise_density'] = noise_density
            temp_df.loc[idx, 'multi_peak_density'] = multi_peak_density
            
            # Add normalized probabilities
            if total_density > 0:
                temp_df.loc[idx, 'noise_probability'] = noise_density / total_density
                temp_df.loc[idx, 'multi_peak_probability'] = multi_peak_density / total_density
            else:
                temp_df.loc[idx, 'noise_probability'] = np.nan
                temp_df.loc[idx, 'multi_peak_probability'] = np.nan
            
            # Add statistical context based on sampling
            if noise_indices and 'noise' in log_density_distributions:
                # Calculate log sum of exp for noise components for this point
                noise_log_density = logsumexp(log_density[noise_indices])
                
                # Calculate percentile in the noise distribution
                noise_percentile = np.mean(log_density_distributions['noise'] < noise_log_density) * 100
                temp_df.loc[idx, 'noise_percentile'] = noise_percentile
                
                noise_weight = sum(model_result['weights'][i] for i in noise_indices) / len(noise_indices)
                
                # Create composite score using the same formula
                # The term (noise_percentile/50) scales the percentile to center around 1.0
                noise_composite_score = (noise_percentile/50) * (noise_weight**0.5) * 100
                
                # Assign confidence based on composite score using same thresholds
                if noise_composite_score > 45:  # Threshold for high confidence
                    temp_df.loc[idx, 'noise_confidence'] = 'high'
                elif noise_composite_score > 25:  # Threshold for medium confidence
                    temp_df.loc[idx, 'noise_confidence'] = 'medium'
                else:
                    temp_df.loc[idx, 'noise_confidence'] = 'low'
            
            if multi_peak_indices and 'multi_peak' in log_density_distributions:
                # Calculate log sum of exp for multi-peak components for this point
                multi_peak_log_density = logsumexp(log_density[multi_peak_indices])
                
                # Calculate percentile in the multi-peak distribution
                multi_peak_percentile = np.mean(log_density_distributions['multi_peak'] < multi_peak_log_density) * 100
                temp_df.loc[idx, 'multi_peak_percentile'] = multi_peak_percentile
                
                # Get component weight for multi-peak components (average if multiple)
                multi_peak_weight = sum(model_result['weights'][i] for i in multi_peak_indices) / len(multi_peak_indices)

                # Create composite score (scales from 0-100)
                # The term (multi_peak_percentile/50) scales the percentile to center around 1.0
                composite_score = (multi_peak_percentile/50) * (multi_peak_weight**0.5) * 100

                # Assign confidence based on composite score
                if composite_score > 60:  # Adjusted thresholds for composite score
                    temp_df.loc[idx, 'multi_peak_confidence'] = 'high'
                elif composite_score > 30:
                    temp_df.loc[idx, 'multi_peak_confidence'] = 'medium'
                else:
                    temp_df.loc[idx, 'multi_peak_confidence'] = 'low'

        # Add to result
        rows_to_include.append(temp_df)
    
    # Combine all results
    if rows_to_include:
        result_df = pd.concat(rows_to_include)
        
        # Sort by p-value (most significant first)
        result_df = result_df.sort_values('p_value')
    
    return result_df


def GMM_analysis(data: pd.DataFrame, model_type: str, out_dir: str, timepoint: str,
                 weight_noise_threshold=0.25, filter_noise=False, 
                 noise_probability_threshold=0.9, plot=True):
    """
    Perform Gaussian Mixture Model analysis on tabular data, save results, and optionally filter noise.
    
    This function fits GMMs to each feature in the provided DataFrame, evaluates and visualizes 
    the results, determines component probabilities for all data points, and saves the results
    to the specified output directory. Optionally, it can filter out noise components.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input data to analyze, with features as columns.
    
    model_type : str
        Type of GMM to use. Options are 'regular', 'bayesian', or 'both'.
        
    out_dir : str
        Path to the output directory where results will be saved.
        
    timepoint : str
        Identifier for the current analysis, used to create a subdirectory.
        
    weight_noise_threshold : float, default=0.25
        Threshold for classifying a component as noise based on its weight.
        
    filter_noise : bool, default=False
        Whether to filter out noise components from the data.
        
    noise_probability_threshold : float, default=0.9
        Threshold probability for classifying a data point as noise when filtering.
    
    Returns
    -------
    tuple
        If filter_noise is True:
            (results_dict, summary_df, all_probs, filtered_data, changed_points_df)
            - results_dict: Dictionary containing GMM analysis results for each feature
            - summary_df: DataFrame summarizing the GMM analysis results
            - all_probs: DataFrame containing component probabilities for each data point
            - filtered_data: DataFrame with noise points removed
            - changed_points_df: DataFrame identifying the filtered data points
            
        If filter_noise is False:
            (results_dict, summary_df, all_probs)
            - results_dict: Dictionary containing GMM analysis results for each feature
            - summary_df: DataFrame summarizing the GMM analysis results
            - all_probs: DataFrame containing component probabilities for each data point
    
    Notes
    -----
    The function creates the following files in the output subdirectory:
    - 'gmm_summary_df.csv': Summary of GMM analysis results
    - 'component_probabilities.csv': Component probabilities for each data point
    
    If filter_noise is True, it also creates:
    - 'noise_filtered_data.csv': Data with noise points removed
    - 'noise_changed_datapoints.csv': Information about removed data points
    """
        
    # Create GMM subdirectory according to timepoint
    sub_dir = os.path.join(out_dir, timepoint)
    os.makedirs(sub_dir, exist_ok=True)

    # Fit GMMs for each feature
    results_dict, summary_df = analyze_tabular_data(data, model_type, 
        weight_noise_threshold,
        )
    # Evaluate and visualize results
    evaluate_results(results_dict, sub_dir, plot)

    # Determine component probabilities for all datapoints
    all_probs = get_component_probabilities(results_dict, data, model_type)

    # Write results to file
    summary_df.to_csv(sub_dir + '/gmm_summary_df.csv')
    all_probs.to_csv(sub_dir + '/component_probabilities.csv')

    if filter_noise:
        # Filter noise datapoints with noise probability above threshold (0.9)
        filtered_data = filter_noise_components(data, all_probs, noise_probability_threshold)

        # Get index and col of removed datapoints
        changed_points_df = get_filtered_points(data, filtered_data)

        # Write to file
        filtered_data.to_csv(sub_dir + '/noise_filtered_data.csv')
        changed_points_df.to_csv(sub_dir + '/noise_changed_datapoints.csv')
        return(results_dict, summary_df, all_probs, filtered_data, changed_points_df)

    else:
        return(results_dict, summary_df, all_probs)



def mode_probabilities(results_dict: Dict, df_significance: pd.DataFrame, df_pat: pd.DataFrame,
                       model_type: str, probability_threshold: float, significance_threshold: float = 0.05):
    """
    Calculate component probabilities for a mixture model and identify points of interest.
    
    This function:
    1. Calculates component/mode probabilities for statistically significant points
    2. Identifies potential artifacts (high noise probability)
    3. Identifies potential interesting cases (high multiple peak probability)
    
    Parameters:
    -----------
    results_dict : Dict
        Dictionary containing model results data, expected to be compatible with the
        calculate_mode_probabilities function.
    df_significance : pd.DataFrame
        DataFrame containing significance metrics, including p-values.
    df_pat : pd.DataFrame
        DataFrame containing patient or sample data used for probability calculations.
    model_type : str
        Type of mixture model used (e.g., 'GMM', 'BMM'). Passed to calculate_mode_probabilities.
    probability_threshold : float
        Threshold value (between 0 and 1) for identifying high-probability noise or 
        multi-peak candidates.
    significance_threshold : float
        Threshold value (between 0 and 1) for identifying significant features.  

    Returns:
    --------
    tuple
        A tuple containing three DataFrames:
        - mode_probs: Complete DataFrame with component probabilities for all significant points
        - noise_candidates: Subset of points with noise probability above the threshold
        - multi_peak_candidates: Subset of points with multi-peak probability above the threshold
    """
    # Calculate component probabilities for significant points
    mode_probs = calculate_mode_probabilities(results_dict, df_significance, df_pat,
                                            significance_threshold=significance_threshold,
                                            model_type=model_type)

    # Examine points with high noise probability (potential artifacts)
    noise_candidates = mode_probs[mode_probs['noise_probability'] > probability_threshold]

    # Examine points with high multiple peak probability (potential interesting cases)
    multi_peak_candidates = mode_probs[mode_probs['multi_peak_probability'] > probability_threshold]

    return mode_probs, noise_candidates, multi_peak_candidates

