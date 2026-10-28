#Imports

# 1. Standard library
import os

# 2. Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import stats


# Functions

def add_patient_id(df):
    """
    Add a patient ID column derived from sample index values.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with a 'Sample_number' column containing sample IDs

    Returns:
    --------
    pd.DataFrame
        Copy of input DataFrame with an additional 'Patient' column
    """
    df = df.copy()
    IDs = []
    for i in df["Sample_number"]:
        ID = i.split("_")[0] + '_' + i.split('_')[1]
        IDs.append(ID)

    df["Patient"] = IDs
    return df



def replicate_mean(df):
    """
    Calculate the mean across replicates for each patient.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing replicate data

    Returns:
    --------
    pd.DataFrame
        DataFrame with per-patient replicate means
    """
    df = df.copy()
    df["Sample_number"] = df.index
    df = add_patient_id(df)
    del df["Sample_number"]

    means = []
    patients = []
    for i in set(df["Patient"]):
        sub_df = df.filter(like=i, axis=0)
        del sub_df["Patient"]
        mean_df = sub_df.mean()
        means.append(mean_df)
        patients.append(i)

    df_means = pd.DataFrame(means)
    df_means["Patients"] = patients
    df_means = df_means.set_index("Patients")

    return df_means



def replicate_median(df):
    """
    Calculate the median across replicates for each patient.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing replicate data

    Returns:
    --------
    pd.DataFrame
        DataFrame with per-patient replicate medians
    """
    df = df.copy()
    df["Sample_number"] = df.index
    df = add_patient_id(df)
    del df["Sample_number"]

    medians = []
    patients = []
    for i in set(df["Patient"]):
        sub_df = df.filter(like=i, axis=0)
        del sub_df["Patient"]
        mean_df = sub_df.median()
        medians.append(mean_df)
        patients.append(i)

    df_medians = pd.DataFrame(medians)
    df_medians["Patients"] = patients
    df_medians = df_medians.set_index("Patients")

    return df_medians



def replicate_var(df):
    """
    Calculate the variance across replicates for each patient.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing replicate data

    Returns:
    --------
    pd.DataFrame
        DataFrame with per-patient replicate variances
    """
    df = df.copy()
    df["Sample_number"] = df.index
    df = add_patient_id(df)
    del df["Sample_number"]

    variances = []
    patients = []
    for i in set(df["Patient"]):
        sub_df = df.filter(like=i, axis=0)
        del sub_df["Patient"]
        var_df = sub_df.var()
        variances.append(var_df)
        patients.append(i)

    df_variances = pd.DataFrame(variances)
    df_variances["Patients"] = patients
    df_variances = df_variances.set_index("Patients")

    return df_variances



def replicate_std(df):
    """
    Calculate the standard deviation across replicates for each patient.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing replicate data

    Returns:
    --------
    pd.DataFrame
        DataFrame with per-patient replicate standard deviations
    """
    df = df.copy()
    df["Sample_number"] = df.index
    df = add_patient_id(df)
    del df["Sample_number"]

    medians = []
    patients = []
    for i in set(df["Patient"]):
        sub_df = df.filter(like=i, axis=0)
        del sub_df["Patient"]
        mean_df = sub_df.std()
        medians.append(mean_df)
        patients.append(i)

    df_medians = pd.DataFrame(medians)
    df_medians["Patients"] = patients
    df_std = df_medians.set_index("Patients")
    

    return df_std



def basic_stats(df_ctrl, df_pat):
    """
    Calculate descriptive statistics and z-scores for patient data relative to controls.

    Parameters:
    -----------
    df_ctrl : pd.DataFrame
        Control data
    df_pat : pd.DataFrame
        Patient data

    Returns:
    --------
    tuple
        - dfp_means : pd.DataFrame — per-patient replicate means
        - dfp_medians : pd.DataFrame — per-patient replicate medians
        - dfp_var : pd.DataFrame — per-patient replicate variances
        - dfp_std : pd.DataFrame — per-patient replicate standard deviations
        - df_z : pd.DataFrame — z-scores relative to control mean and std
    """
    dfc = df_ctrl.copy()
    dfp = df_pat.copy()

    ### Gather Control Statistics
    c_mean = []
    c_std = []
    for column in dfc.columns:
        c_mean.append(np.mean(dfc[column]))
        c_std.append(np.std(dfc[column], ddof=1))

    ### Gather Patient Replicate Statistics
    dfp_means = replicate_mean(dfp.copy())
    dfp_medians = replicate_median(dfp.copy())
    dfp_var = replicate_var(dfp.copy())
    dfp_std = replicate_std(dfp.copy())

    ### Calculate z-scores -> Distributions often not normal
    z_score = []
    for i in range(len(dfp_means)):
        sub_z = []
        for j in range(dfp_means.shape[1]):
            z = (dfp_means.iloc[i, j] - c_mean[j]) / c_std[j]
            sub_z.append(z)
        z_score.append(sub_z)
    df_z = pd.DataFrame(z_score)
    df_z.columns = dfp_means.columns
    df_z.index = dfp_means.index

    ### Collect in list
    return(dfp_means, dfp_medians, dfp_var, dfp_std, df_z)



def permutation_pval(df_ctrl, df_pat, mt_correction):
    """
    Estimate p-values of observations from their rank in the control distribution.

    Parameters:
    -----------
    df_ctrl : pd.DataFrame
        Control data (Cohort 1)
    df_pat : pd.DataFrame
        Patient data
    mt_correction : str
        Multiple testing correction method: 'Bonferroni', 'FDR', or ''

    Returns:
    --------
    df_pvals_corrected : pd.DataFrame
        Corrected p-values (patients × features)
    """
    df_pvals = pd.DataFrame(index=df_pat.index)
    # for each feature
    for col in df_ctrl.columns:
        ctrl_feat_m = list(df_ctrl.loc[:, col])
        ctrl_ind = list(df_ctrl.index)
        feature_pvals = []
        # calculate feature pvals for each patient
        for ind in df_pat.index:
            pat_feat_m = [df_pat.loc[ind, col]]

            # append pat value to list of ctrl values
            rank_list = ctrl_feat_m + pat_feat_m
            indices = ctrl_ind + [ind]
            rank_df = pd.DataFrame(rank_list, index=indices)

            # get ranks
            rank_df['Rank'] = rank_df.rank()
            pat_rank = rank_df.loc[ind, 'Rank']

            # calculate two sided rank
            rank_low = pat_rank
            rank_high = len(rank_df) - pat_rank + 1
            k = min(rank_low, rank_high)  # k = relevant rank

            # two-sided p-value 
            pval = 2 * k / (len(rank_df) + 1)
            feature_pvals.append(pval)
        df_pvals[col] = feature_pvals

    # correct for multiple testing
    if mt_correction.lower() == 'bonferroni':
        df_pvals_corrected = df_pvals * df_pvals.shape[1]
        df_pvals_corrected = df_pvals_corrected.clip(upper=1)

    elif mt_correction == 'FDR':
        df_pvals_corrected = stats.false_discovery_control(df_pvals, axis=1)
    else:
        df_pvals_corrected = df_pvals
    return(df_pvals_corrected)



def calculate_pvals(df_ctrl, dfp_means, dfp_medians, mode, p_fxn, mt_correction):
    """
    Calculate p-values for each feature using the specified method.

    Parameters:
    -----------
    df_ctrl : pd.DataFrame
        Control data
    dfp_means : pd.DataFrame
        Per-patient replicate means
    dfp_medians : pd.DataFrame
        Per-patient replicate medians
    mode : str
        Replicate summary statistic to use: 'mean', 'median', or 'average'
    p_fxn : str
        P-value estimation method: 'permutation'
    mt_correction : str
        Multiple testing correction method: 'Bonferroni', 'FDR', or ''

    Returns:
    --------
    tuple
        - df_pval : pd.DataFrame — p-values
        - df_p_inv : pd.DataFrame — inverse p-values
        - df_p_val_size : pd.DataFrame — significance levels (0=ns, 1=0.05, 2=0.0001, 3=1e-10)
        - df_p_zero : pd.DataFrame — -log10(p-value) + 1
    """
    ### Generate copies of all dfs
    dfc = df_ctrl.copy()
    dfp_means = dfp_means.copy()
    dfp_medians = dfp_medians.copy()

    ###### Empirical p-values
    ### Calculate p-values
    # list: 1 x #features, arrays: #samples x 1
    if mode == 'mean':
        if p_fxn == 'permutation':
            df_pval = permutation_pval(dfc, dfp_means, mt_correction)
        else:
            print('Chose a valid method for p-value estimation.')
    elif mode == 'median':
        if p_fxn == 'permutation':
            df_pval = permutation_pval(dfc, dfp_medians, mt_correction)
        else:
            print('Chose a valid method for p-value estimation.')
    elif mode == 'average':
        if p_fxn == 'permutation':
            df_pval = permutation_pval(dfc, dfp_means, mt_correction)
            df_pval = replicate_mean(df_pval)
        else:
            print('Chose a valid method for p-value estimation.')
    else:
        print('Invalid mode.')

    # transform to df
    df_pval = pd.DataFrame(df_pval)
    df_pval.columns = dfp_means.columns
    df_pval.index = dfp_means.index

    # replace 0 p-vals by min float
    df_pval = df_pval.replace(0, 2.2e-16)

    ### Calculate inverse p-values
    df_p_inv = 1 / df_pval

    ### Calculate p_val_size
    p_val_size = []
    for row in df_pval.index:
        rowlist = []
        for col in df_pval.columns:
            if df_pval.loc[row, col] <= 0.0000000001:
                rowlist.append(3)
            elif df_pval.loc[row, col] <= 0.0001:
                rowlist.append(2)
            elif df_pval.loc[row, col] <= 0.05:
                rowlist.append(1)
            else:
                rowlist.append(0)
        p_val_size.append(rowlist)

    df_p_val_size = pd.DataFrame(p_val_size)
    df_p_val_size.columns = dfp_means.columns
    df_p_val_size.index = dfp_means.index
    df_p_val_size = df_p_val_size.astype(int)

    ### p_val zeroes
    df_p_zero = -np.log10(df_pval) + 1
    df_p_zero = df_p_zero.replace(np.inf, 10000).astype(int)
    df_p_zero = df_p_zero.replace(10000, np.inf)

    return(df_pval, df_p_inv, df_p_val_size, df_p_zero)



def correct_gmm_pvalues(df_significance, mt_correction):
    """
    Apply multiple testing correction to the GMM p-value columns.

    Correction is applied per patient (row-wise, across features).
    Missing p-values are set to 1 before correction.

    Parameters:
    -----------
    df_significance : pd.DataFrame
        GMM significance table with '<feature>_pvalue' columns, as returned by
        calculate_gmm_pvalues_for_triplicates
    mt_correction : str
        Multiple testing correction method: 'Bonferroni', 'FDR', or '' (no correction)

    Returns:
    --------
    pd.DataFrame
        Copy of df_significance with corrected '<feature>_pvalue' columns
    """
    df_significance = df_significance.copy()
    cols = df_significance.columns[df_significance.columns.str.endswith('_pvalue')]
    pvals = df_significance[cols].fillna(1)

    if mt_correction.lower() == 'fdr':
        df_significance[cols] = stats.false_discovery_control(pvals, axis=1)
    elif mt_correction.lower() == 'bonferroni':
        df_significance[cols] = (pvals * len(cols)).clip(upper=1)
    else:
        df_significance[cols] = pvals

    return df_significance



def stats_pvals_gmm(df_pval, dfp_means):
    """
    Compute derived p-value statistics from a pre-calculated p-value DataFrame.

    Parameters:
    -----------
    df_pval : pd.DataFrame
        P-values per sample and feature
    dfp_means : pd.DataFrame
        Per-patient replicate means, used for column/index alignment

    Returns:
    --------
    tuple
        - df_pval : pd.DataFrame — p-values with zeros replaced by 2.2e-16
        - df_p_inv : pd.DataFrame — inverse p-values
        - df_p_val_size : pd.DataFrame — significance levels (0=ns, 1=0.05, 2=0.0001, 3=1e-10)
        - df_p_zero : pd.DataFrame — -log10(p-value) + 1
    """

    # replace 0 p-vals by min float
    df_pval = df_pval.replace(0, 2.2e-16)

    ### Calculate inverse p-values
    df_p_inv = 1 / df_pval
    df_p_inv.columns = dfp_means.columns

    ### Calculate p_val_size
    p_val_size = []
    for row in df_pval.index:
        rowlist = []
        for col in df_pval.columns:
            if df_pval.loc[row, col] <= 0.0000000001:
                rowlist.append(3)
            elif df_pval.loc[row, col] <= 0.0001:
                rowlist.append(2)
            elif df_pval.loc[row, col] <= 0.05:
                rowlist.append(1)
            else:
                rowlist.append(0)
        p_val_size.append(rowlist)

    df_p_val_size = pd.DataFrame(p_val_size)
    df_p_val_size.columns = dfp_means.columns
    df_p_val_size.index = df_pval.index
    df_p_val_size = df_p_val_size.astype(int)

    ### p_val zeroes
    df_p_zero = -np.log10(df_pval) + 1
    df_p_zero = df_p_zero.replace(np.inf, 10000).astype(int)
    df_p_zero = df_p_zero.replace(10000, np.inf)
    df_p_zero.columns = dfp_means.columns

    return(df_pval, df_p_inv, df_p_val_size, df_p_zero)



def write_stats(df_FC_pat, mode_probabilities_2, df_p_inv, df_p_val_size, df_p_zero, df_z, sig_level, output_path, ctrl_nan, nan_filter, fc_crop, df_pval = None):
    """
    Write per-patient statistics to CSV and generate result overview plots.

    For each patient, compiles fold changes, p-values, z-scores, and GMM mode
    probabilities into a Stats.csv file and a scatter plot overview.

    Parameters:
    -----------
    df_FC_pat : pd.DataFrame
        Fold changes for patient samples
    mode_probabilities_2 : pd.DataFrame
        GMM mode probabilities of processed data
    df_p_inv : pd.DataFrame
        Inverse p-values
    df_p_val_size : pd.DataFrame
        Significance level encodings
    df_p_zero : pd.DataFrame
        -log10(p-value) + 1 values
    df_z : pd.DataFrame
        Z-scores
    sig_level : float
        P-value threshold for significance
    output_path : str
        Root output directory path
    ctrl_nan : pd.Series
        NaN proportions per feature in the control cohort
    nan_filter : float
        Maximum allowed NaN proportion for feature inclusion
    fc_crop : float
        Lower bound for log2 fold change cropping
    df_pval : pd.DataFrame, optional
        Pre-calculated p-values; if None, p-values are taken from mode_probabilities_2

    Returns:
    --------
    pd.DataFrame
        Boolean DataFrame indicating significant features per patient
    """
    print('Writing Stats.csv and Statsplots.')
    # Make copies to avoid editing dataframes
    df_FC_pat = df_FC_pat.copy()
    mode_probabilities_2 = mode_probabilities_2.copy()
    df_p_inv = df_p_inv.copy()
    df_p_val_size = df_p_val_size.copy()
    df_p_zero = df_p_zero.copy()
    df_z = df_z.copy()
    
    ### For each Patient:
    pat_id_set = list(set(mode_probabilities_2.index))
    # Intialize list of lists for significant features
    sig_lol = []
    for i in pat_id_set:
        # Process mode_probabilities - GMM 2
        df_modes_2 = mode_probabilities_2.loc[mode_probabilities_2.index.str.contains(i)]
        df_modes_2 = df_modes_2.sort_values(by='feature')
        df_modes_2.index = df_modes_2['feature']

        noise_density = df_modes_2.noise_density
        multipeak_density = df_modes_2.multi_peak_density
        # Add new confidence metrics
        noise_confidence = df_modes_2.noise_confidence if 'noise_confidence' in df_modes_2.columns else pd.Series('unknown', index=df_modes_2.index)
        multi_peak_confidence = df_modes_2.multi_peak_confidence if 'multi_peak_confidence' in df_modes_2.columns else pd.Series('unknown', index=df_modes_2.index)
        
        # FC cleanup
        df_FC_rep = df_FC_pat.loc[df_FC_pat.index.str.contains(i)]
        del df_FC_rep["Class"]
        FC = df_FC_rep.mean()
        FC_log2 = np.log2(FC)
        # save uncropped FC_log2
        FC_log2_uncropped = FC_log2.copy()
        # crop negative FC_log2 at -3
        FC_log2[FC_log2 < fc_crop] = fc_crop

        samplename = df_FC_rep.index[0].split("_")[0] + '_' + df_FC_rep.index[0].split('_')[1]
        sample_out = output_path + "Patients/" + samplename
        # Extract columns of interest and transpose
        if df_pval is not None:
            pvals = df_pval.loc[samplename]
            # Clean 0 and inf values
            pvals = pvals.replace(0, 2e-16)
            log10_p = -np.log10(pvals)

        else:
            pvals = df_modes_2.p_value
            # Clean 0 and inf values
            pvals = pvals.replace(0, 2e-16)
            log10_p = -np.log10(pvals)

        # Gather Statistics in df_stats and write to csv
        arr = [
            FC,
            FC_log2,
            FC_log2_uncropped,
            pvals,
            df_p_inv.loc[samplename],
            df_p_val_size.loc[samplename],
            df_p_zero.loc[samplename],
            df_z.loc[samplename],
            log10_p,
            noise_density,
            noise_confidence,
            multipeak_density,
            multi_peak_confidence
        ]

        arrl = (
            "FC",
            "FC_log2",
            "FC_log2_uncropped",
            "p_value",
            "p_value_inverse",
            "p_val_size",
            "p_value_zeroes",
            "z-score",
            "-log10P",
            "noise_density",
            "noise_confidence",
            "multipeak_density",
            "multi_peak_confidence"
        )

        df_stats = pd.DataFrame(arr, columns=df_FC_rep.columns, index=arrl)
        
        # gather significant features for plotting
        sig_features = df_stats.loc['p_value'] <= sig_level
        sig_lol.append(sig_features)
        
        ### Exclude df_stats features with nan proportion > nan_filter
        # get column names with nan proportion > nan_filter
        remove_features = [ctrl_nan.index[i] for i in range(len(ctrl_nan)) if ctrl_nan.iloc[i] > nan_filter]
        # remove IS labels (IS have numbers >= 1000)
        remove_features = [x for x in remove_features if len(x.split('_')[0]) < 4]
        # if significant, remove feature from remove_features
        remove_features = [x for x in remove_features if df_stats.loc['p_value',x] > sig_level]
        df_stats = df_stats.drop(columns=remove_features)

        # if output dir does not exists (ommited FC plots) generate the dir
        if not os.path.exists(sample_out):
            os.makedirs(sample_out)
        df_stats.to_csv(sample_out + "/Stats.csv")

        # Generate Statsplot
        sns.set_theme()
        df1 = df_stats.T
        df2 = df1.iloc[:, :]
        
        # Extract significant features and add feature classification
        df2['feature_type'] = 'normal'
        significant_mask = df2['p_value'] < 1
        df2.loc[significant_mask & (df2['noise_confidence'] == 'high'), 'feature_type'] = 'noise'
        df2.loc[significant_mask & 
                (df2['multi_peak_confidence'] == 'high') & 
                (df2['FC_log2'] >= -1), 'feature_type'] = 'multipeak'

        # Calculate point sizes
        xsize = df2.p_value_zeroes
        xsize = 5 + xsize*1.1
        xa = list(df2.index)
        xb = list(df2["FC"])

        # Create figure with explicit axes
        fig, ax = plt.subplots(figsize=(14, 6))
        plt.subplots_adjust(right=0.65)  # Leave more space on the right
        # Instead of using scatterplot directly, we'll create our own plot
        # Create consistent size mapping between feature points and highlighted points

        
        # Define a function to map p_value_size to point sizes consistently
        def size_mapping(p_val_size):
            if p_val_size == 0:
                return(20)
            elif p_val_size == 1:
                return(40)
            elif p_val_size == 2:
                return(60)
            elif p_val_size == 3:
                return(80)
            

        # Create main plot manually with consistent size mapping
        g = plt.scatter(
            range(len(xa)), 
            df2['FC_log2'],
            c=df2['-log10P'],  # Use p_val_size for color

            s=[size_mapping(x) for x in df2['p_val_size']],  # Consistent size mapping
            cmap='coolwarm',
            alpha=0.7,
            vmin=0,
            vmax=10
        )

        # Now overlay our highlighted points with the SAME size mapping
        for j, (idx, row) in enumerate(df2.iterrows()):
            if row['feature_type'] == 'noise':
                plt.scatter(
                    j, 
                    row['FC_log2'],
                    s=size_mapping(row['p_val_size']),
                    color='green',
                    linewidth=1,
                    zorder=10
                )
            elif row['feature_type'] == 'multipeak':
                plt.scatter(
                    j, 
                    row['FC_log2'],
                    s=size_mapping(row['p_val_size']),
                    color='purple',
                    linewidth=1,
                    zorder=10
                )
        
        # Set up plot parameters
        plt.margins(x=0.05)
        plt.title(i)
        plt.ylabel("Fold Change (log2)")
        plt.xlabel("Compound")
        plt.xticks(range(len(xa)), ['' for _ in xa])  # Empty tick labels
        plt.axhline(0, ls="--", color="black", linewidth=0.75)
        plt.axhline(1, ls="--", color="black", linewidth=0.5)
        plt.axhline(-1, ls="--", color="black", linewidth=0.5)
        
        # Legend for feature properties
        has_noise = 'noise' in df2['feature_type'].values
        has_multipeak = 'multipeak' in df2['feature_type'].values

        property_handles = []
        property_labels = []
        
        if 'noise' in df2['feature_type'].values:
            property_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8))
            property_labels.append('Noise (High Confidence)')

        if 'multipeak' in df2['feature_type'].values:
            property_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8))
            property_labels.append('Multipeak (High Confidence)')

        # Position feature legend inside the plot at top-right
        if property_handles:
            # First, place the legend to check if we need more space
            legend = ax.legend(
                property_handles,
                property_labels,
                title="Feature Properties",
                title_fontsize="small",
                fontsize="x-small",
                loc="upper right",
                frameon=True,  # Add a white background
                fancybox=True,
                framealpha=0.9  # Slightly transparent background
            )
            
            # Extend y-axis if needed to make room for legend
            current_ymin, current_ymax = ax.get_ylim()
            
            # Calculate extra space needed for legend (roughly 0.5-1 unit per legend entry)
            extra_space = 0.7 * len(property_labels)
            
            # Set new y-limits with extra space at top for legend
            ax.set_ylim(min(current_ymin, -4), max(current_ymax + extra_space, 4 + extra_space))

        plt.tight_layout()
        plt.savefig(sample_out + "/result_overview.png", bbox_inches='tight')
        plt.close()

    # Gather sig_lol in dataframe
    sig_lol = pd.DataFrame(sig_lol, index=pat_id_set)

    return(sig_lol)


