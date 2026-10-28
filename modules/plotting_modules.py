# Imports

# 1. Standard library
import os

# 2. Third-party
import gc
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')

# Set seaborn theme and palette
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
def prepare_data_for_pca(df):
    """
    Prepare the data for PCA by identifying cohorts and filtering out QC samples.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sample IDs as index and features as columns
        
    Returns:
    --------
    tuple
        (Filtered DataFrame with features, Series with cohort labels)
    """
    # Extract sample IDs from the DataFrame index
    sample_ids = df.index.tolist()
    
    # Identify cohort based on the first character of sample ID
    cohorts = []
    valid_indices = []
    
    for i, sample_id in enumerate(sample_ids):
        if str(sample_id).startswith('1'):
            cohorts.append('Cohort 1')
            valid_indices.append(i)
        elif str(sample_id).startswith('2') or str(sample_id).startswith('3'):
            cohorts.append('Cohort 2')
            valid_indices.append(i)
        elif str(sample_id).startswith('M4D'):
            # Skip QC samples
            continue
        else:
            # Skip unknown samples
            continue
    
    # Filter the DataFrame to include only Cohort 1 and Cohort 2 samples
    filtered_df = df.iloc[valid_indices]

    # Impute nan as halfmin
    filtered_df = filtered_df.fillna(filtered_df.min().min()/2)
    # Create a Series with cohort labels
    cohort_labels = pd.Series(cohorts, index=filtered_df.index)
    
    return filtered_df, cohort_labels



def perform_pca(df):
    """
    Perform PCA on the data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
        
    Returns:
    --------
    tuple
        (PCA components DataFrame, PCA model, Explained variance ratios)
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Perform PCA
    pca = PCA(n_components=3)
    components = pca.fit_transform(scaled_data)
    
    # Create a DataFrame with the PCA components
    pca_df = pd.DataFrame(
        data=components,
        columns=['PC1', 'PC2', 'PC3'],
        index=df.index
    )
    
    return pca_df, pca, pca.explained_variance_ratio_



def plot_3d_pca(pca_df, cohort_labels, explained_variance, save_path=None):
    """
    Create a 3D PCA scatter plot coloured by cohort.

    Parameters:
    -----------
    pca_df : pd.DataFrame
        DataFrame with PC1, PC2, PC3 columns, as returned by perform_pca
    cohort_labels : pd.Series
        Series mapping sample IDs to cohort labels, as returned by prepare_data_for_pca
    explained_variance : np.ndarray
        Explained variance ratios for PC1, PC2, PC3, as returned by perform_pca
    save_path : str, optional
        Directory path where the plot will be saved. If None, plot is not saved.

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Create a color map for the cohorts
    colors = sns.color_palette("deep")

    # Create mapping dictionaries for cohort names and colors
    cohort_colors = {'Cohort 1': colors[0], 'Cohort 2': colors[3]}
    cohort_display_names = {'Cohort 1': 'Control Cohort', 'Cohort 2': 'Patient Cohort'}
    
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')  # Set the axes background to white
    
    # Plot each cohort
    for cohort in ['Cohort 1', 'Cohort 2']:
        # Filter data for the current cohort
        cohort_data = pca_df[cohort_labels == cohort]
        
        # Plot the cohort data with the display name in the legend
        ax.scatter(
            cohort_data['PC1'],
            cohort_data['PC2'],
            cohort_data['PC3'],
            color=cohort_colors[cohort],
            label=cohort_display_names[cohort],  # Use the display name here
            alpha=0.7,
            edgecolors='w',
            s=80
        )
    
    # Set labels and title
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
    ax.set_zlabel(f'PC3 ({explained_variance[2]:.1%})')
    ax.set_title('3-D PCA of Control and Sample Cohorts')
    # Remove ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # Hide tick marks by making them transparent
    ax.tick_params(axis='x', which='both', colors=(0,0,0,0))
    ax.tick_params(axis='y', which='both', colors=(0,0,0,0))
    ax.tick_params(axis='z', which='both', colors=(0,0,0,0))
    ax.legend(title='Cohort')
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path is not None:
        plt.savefig(save_path + '/Sample_PCA.png', dpi=300)
        
    return fig



def run_pca_analysis(df, save_path=None):
    """
    Run the complete PCA analysis workflow.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sample IDs as index and features as columns
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns:
    --------
    tuple
        (PCA components DataFrame, Figure object)
    """
    # Prepare the data
    filtered_df, cohort_labels = prepare_data_for_pca(df)
    
    # Perform PCA
    pca_df, pca, explained_var = perform_pca(filtered_df)
    
    # Create the plot
    fig = plot_3d_pca(pca_df, cohort_labels, explained_var, save_path)
    
    # Show the plot
    plt.show()
    
    return pca_df, fig



def plot_FC_violin(pat_id_set, df_FC_pat, df_FC_ctrl, output_path, write, sig_lol):
    """
    Generate violin plots for fold changes of significant metabolites per patient.

    Parameters:
    -----------
    pat_id_set : list
        List of patient IDs
    df_FC_pat : pd.DataFrame
        Fold changes of patient data
    df_FC_ctrl : pd.DataFrame
        Fold changes of control data
    output_path : str
        Output directory path
    write : bool
        If True, generates and saves plots; if False, skips plotting
    sig_lol : pd.DataFrame
        Boolean DataFrame indicating significant metabolites per patient
    """
    # get df_FC for each patient
    if write:
        print('Plotting FC violin plots.')

        for i in pat_id_set:
            #print(i)
            df_FC_rep = df_FC_pat.loc[df_FC_pat.index.str.contains(i)]
            df_FC_conc = pd.concat([df_FC_ctrl, df_FC_rep])

            folder = output_path + 'Patients/' + i 
            os.makedirs(folder, exist_ok=True)
            os.makedirs(folder + '/FC/', exist_ok=True)

            for column in df_FC_conc.columns[:-1]:
                # Only generate plots for significant metabolites    
                if sig_lol.loc[i, column] == True:    
                    g = sns.violinplot(x='Class', y=column, hue=df_FC_conc['Class'], legend=False, palette = "Blues_d" ,  data=df_FC_conc, cut=0)
                    title = column.split("_")[1]
                    plt.title(title)
                    plt.ylabel("Fold Change")
                    plt.plot()
                    plt.close()
                    fig_path = folder + '/FC/' + column + '.png'   # full feature name avoids overwriting (e.g. 065_Creatine_pos vs 904_Creatine_Creatinine_ratio)
                    g.figure.savefig(fig_path)

                    plt.clf()
                    plt.close('all')
                    gc.collect()

    else:
        print('Plotting FC is turned off.')



def plot_log_FC_violin(pat_id_set, df_FC_pat, df_FC_ctrl, output_path, write, sig_lol):
    """
    Generate violin plots for log2 fold changes of significant metabolites per patient.

    Parameters:
    -----------
    pat_id_set : list
        List of patient IDs
    df_FC_pat : pd.DataFrame
        Log2 fold changes of patient data
    df_FC_ctrl : pd.DataFrame
        Log2 fold changes of control data
    output_path : str
        Output directory path
    write : bool
        If True, generates and saves plots; if False, skips plotting
    sig_lol : pd.DataFrame
        Boolean DataFrame indicating significant metabolites per patient
    """
    # get df_FC for each patient
    if write:
        print('Plotting log2FC violin plots.')
        for i in pat_id_set:
            df_FC_rep = df_FC_pat.loc[df_FC_pat.index.str.contains(i)]
            df_FC_conc = pd.concat([df_FC_ctrl, df_FC_rep])

            samplename = df_FC_rep.index[0].split('_')[0]

            folder = output_path + 'Patients/' + i 
            os.makedirs(folder, exist_ok=True)
            os.makedirs(folder + '/LogFC/', exist_ok=True)

            for column in df_FC_conc.columns[:-1]:
                # Only generate plots for significant metabolites
                if sig_lol.loc[i, column] == True:       
                    g = sns.violinplot(x='Class', y=column, hue=df_FC_conc['Class'], legend=False, palette="Blues_d", data=df_FC_conc, cut=0)                    
                    g.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    title = column.split("_")[1]
                    plt.title(title)
                    plt.ylabel("Log2 Fold Change")
                    plt.plot()
                    plt.close()
                    fig_path = folder + '/LogFC/' + column + '.png'   # full feature name avoids overwriting
                    g.figure.savefig(fig_path)

                    plt.clf()
                    plt.close('all')
                    gc.collect()

    else:
        print('Plotting log2FC is turned off.')



