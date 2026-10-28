# Imports
# 1. Standard library
# (none)

# 2. Third-party
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
import seaborn as sns

# Set seaborn theme and palette
sns.set_theme(style='white')
# Set global font sizes
plt.rcParams.update({
    'font.size': 14,          # Default text size
    'axes.titlesize': 16,     # Title size
    'axes.labelsize': 14,     # Axis label size
    'xtick.labelsize': 12,    # X-axis tick label size
    'ytick.labelsize': 12,    # Y-axis tick label size
    'legend.fontsize': 12,    # Legend text size
    'figure.titlesize': 18    # Figure title size
})

# Functions

def create_interactive_statsplot(df_stats, output_path=None, show_plot=True, title=None):
    """
    Create an interactive scatter plot of per-patient feature statistics.

    Points are coloured by -log10(p-value) and sized by significance level.
    Hovering over a point displays the feature name via mplcursors.

    Parameters:
    -----------
    df_stats : pd.DataFrame
        Statistics DataFrame in wide format (features as columns), as written
        by write_stats
    output_path : str, optional
        Directory path where the plot will be saved. If None, plot is not saved.
    show_plot : bool, optional
        Whether to display the plot interactively, by default True
    title : str, optional
        Plot title. If None, derived from output_path or defaults to 'Feature Analysis'.

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    
    # Close any existing figures first
    plt.close('all')
    # Create working copy of data
    df2 = df_stats.T.copy()
    
    # Convert relevant columns to numeric types
    numeric_columns = ['FC_log2', 'p_value', 'p_val_size', '-log10P']
    for column in numeric_columns:
        if column in df2.columns:
            df2[column] = pd.to_numeric(df2[column], errors='coerce')

    # Fill any potential NaN values that might result from conversion errors
    df2 = df2.fillna(0)
    
    # Get data values for plotting
    # Convert all index values to strings
    xa = [str(x) for x in df2.index]
    
    # Define consistent size mapping (same as write_stats)
    def size_mapping(p_val_size):
        if p_val_size == 0:
            return 20
        elif p_val_size == 1:
            return 80
        elif p_val_size == 2:
            return 100
        elif p_val_size == 3:
            return 120
        else:
            return 20  # fallback
    
    # Convert p_val_size to numeric values
    df2['p_val_size'] = df2['p_val_size'].astype(int)
    
    # Create a new figure with explicit axes object (matching write_stats layout)
    fig, ax = plt.subplots(figsize=(14, 6))
    plt.subplots_adjust(right=0.65)  # Leave more space on the right (same as write_stats)

    # Convert values to explicit numeric types
    x_numeric = np.arange(len(xa), dtype=float)  
    y_numeric = df2['FC_log2'].astype(float)     

    # Create main scatter plot with -log10P coloring (matching write_stats exactly)
    g = ax.scatter(
        x_numeric,
        y_numeric,
        c=df2['-log10P'],  # Use -log10P for color (same as write_stats)
        s=[size_mapping(x) for x in df2['p_val_size']],  # Consistent size mapping
        cmap='viridis_r',
        alpha=0.7,
        vmin=0,
        vmax=4  # Same color scale as write_stats
    )

    # Store the scatter points for hover functionality 
    all_scatter_points = [g]
    
    # Set up plot parameters (matching write_stats exactly)
    plt.margins(x=0.05)
    plt.title(title if title is not None else (output_path.split('/')[-2] if output_path else "Feature Analysis"))
    plt.ylabel("Fold Change (log2)")
    plt.xlabel("Compound")
    plt.xticks([])  # Completely remove xticks
    plt.axhline(0, ls="--", color="black", linewidth=0.75)
    plt.axhline(1, ls="--", color="black", linewidth=0.5)
    plt.axhline(-1, ls="--", color="black", linewidth=0.5)
    
    # Add colorbar in the same position as write_stats
    cbar = fig.colorbar(g, ax=ax, pad=0.02, fraction=0.046, aspect=20)
    cbar.set_label('-log10(p-value)', rotation=270, labelpad=15)
    
    # Add hover functionality using mplcursors
    cursor = mplcursors.cursor(all_scatter_points, hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        # Get the x-coordinate (integer position)
        x_position = int(sel.target[0])  # Convert to int since it's a position
        # Use that as an index into the feature names
        feature_name = xa[x_position]

        # Set up hover annotation
        sel.annotation.set_text(str(feature_name))  # Ensure it's a string
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
        sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=0.7)
        
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path + "/interactive_statsplot.png", bbox_inches='tight', dpi=150)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return plt.gcf()  # Return the figure for further customization if needed