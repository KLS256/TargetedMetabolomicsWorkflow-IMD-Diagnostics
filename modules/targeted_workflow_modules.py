# Imports

# 1. Standard library
import os
from datetime import datetime

# 2. Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Functions

def input_directories(input_path, sampletype):
    """
    Generate list of input directories filtered by sample type.

    Parameters:
    -----------
    input_path : str
        Path to input directory
    sampletype : str
        Sample type filter: 'U' for urine, 'P' for plasma

    Returns:
    --------
    list
        List of matching input folder names
    """
    dirs = os.listdir(input_path)
    # Filter Input for specified sampletype
    if sampletype == "U":
        dirs = [x for x in dirs if "U" in x]
    elif sampletype == "P":
        dirs = [x for x in dirs if "P" in x]
    else:
        print("Please choose an appropriate sampletype.")

    return dirs



def generate_outdirs(output_path):
    """
    Generate a dated output directory structure for a new run.

    Creates subdirectories for Plasma, Urine, Patients, and intermediate files.

    Parameters:
    -----------
    output_path : str
        Path to the general output folder

    Returns:
    --------
    str
        Path to the newly created run directory
    """
    date = datetime.today().strftime('%Y-%m-%d')
    output = output_path + date

    i=1
    while True:
        out = output + '/Run' + str(i)
        if not os.path.exists(out):
            os.makedirs(out)
            os.makedirs(out + '/Plasma')
            os.makedirs(out + '/Urine')
            os.makedirs(out + '/Patients')
            os.makedirs(out + '/Intermediate_Files_U')
            os.makedirs(out + '/Intermediate_Files_P')
            break
        i += 1
    print('Run ' + str(i) +':')
    print(' ')
    return(out)



def read_MS_data(input_path, folders, df_metabolites):
    """
    Read MS data from selected batch folders.

    Parameters:
    -----------
    input_path : str
        Path to the main input folder
    folders : list
        Folder names for the batches of interest
    df_metabolites : pd.DataFrame
        DataFrame containing targeted metabolite labels

    Returns:
    --------
    tuple
        - df : pd.DataFrame — all sample data from the selected batches
        - batchlist : list — batch labels corresponding to each sample
    """

    df = df_metabolites
    batchlist = []
    # Iterate through folders
    for i in range(len(folders)):
        loc = input_path + folders[i] + "/"

        # Get all file names in folder
        flist = os.listdir(loc)

        for j in range(len(flist)):
            loc2 = loc + flist[j]
            df1 = pd.read_csv(loc2, sep="\t", index_col=0)
            t, rst = df1.columns[0].split(".")
            df1.rename(columns={df1.columns[0]: t}, inplace=True)
            df = df.join(df1, lsuffix="", rsuffix="_drop")
            batchlist.append(folders[i])

    return (df, batchlist)



def drop_blanks(df):
    """
    Remove blank measurements from the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing MS data

    Returns:
    --------
    pd.DataFrame
        MS data without blank measurements
    """
    df = df.loc[:, ~df.columns.str.contains("blank")]
    df = df.loc[:, ~df.columns.str.contains("Blank")]

    return df



def transpose_samples(df_samples):
    """
    Transpose the sample DataFrame and set index name.

    Parameters:
    -----------
    df_samples : pd.DataFrame
        DataFrame containing MS data

    Returns:
    --------
    pd.DataFrame
        Transposed DataFrame with index named '000_Sample_number'
    """
    df_t = df_samples.transpose()
    df_t.index.name = "000_Sample_number"

    return df_t



def append_batch_to_id(df_samples, df_batches):
    """
    Append batch ID to sample IDs to avoid conflicts from duplicate runs.

    Parameters:
    -----------
    df_samples : pd.DataFrame
        DataFrame containing MS data
    df_batches : pd.DataFrame
        DataFrame mapping sample IDs to batch IDs

    Returns:
    --------
    pd.DataFrame
        DataFrame with batch-appended sample IDs
    """
    # batches as list
    batch_list = list(df_batches['Batches'])

    # generate new ids
    new_ids = []
    for i in range(len(df_samples.index)):
        ind = df_samples.index[i]
        batch = batch_list[i]

        # Special case with M4D and drop in ID
        if 'M4D' in ind and 'drop' in ind:
            qc1, qc2, qc3, qc4 = ind.split('_')
            new_ind = qc1 + '_' + qc2 + '_' + batch + '_' + qc3
        
        # Replace 'drop' by batch id
        elif 'drop' in ind:
            sample, pos, drop = ind.split('_')
            new_ind = sample + '_' + batch + '_' + pos

        # Handle special case of QC samples
        elif 'M4D' in ind:
            qc1, qc2, qc3 = ind.split('_')
            new_ind = qc1 + '_' + qc2 + '_' + batch + '_' + qc3

        # Append batch id to unique samples
        else:
            sample ,pos = ind.split('_')
            new_ind = sample + '_' + batch + '_' + pos
        new_ids.append(new_ind)

    # Replace sample ids
    df_samples.index = new_ids
    df_samples.index.name = '000_Sample_number'

    return(df_samples)




def merge_modes(df_samples):
    """
    Merge positive and negative ionization mode data per sample.

    Parameters:
    -----------
    df_samples : pd.DataFrame
        DataFrame containing MS data with '_pos' and '_neg' suffixed columns

    Returns:
    --------
    pd.DataFrame
        DataFrame with positive and negative mode measurements merged per sample
    """

    # create df's for positive & negative modes
    df_pos = df_samples.loc[:, df_samples.columns.str.contains("_pos")]
    df_pos = df_pos.loc[df_pos.index.str.contains("_pos")]
    df_neg = df_samples.loc[:, df_samples.columns.str.contains("_neg")]
    df_neg = df_neg.loc[df_neg.index.str.contains("_neg")]

    # remove pos & neg from index
    pos_index = [df_pos.index.str.replace("_pos", "_")]
    neg_index = [df_neg.index.str.replace("_neg", "_")]
    df_pos = df_pos.set_index(keys=pos_index)
    df_neg = df_neg.set_index(keys=neg_index)

    # merge pos & neg to get all measurements per sample in 1 row
    df_merged = df_pos.merge(df_neg, on="000_Sample_number")

    # sort by metabolite ID
    indexnumber = [int(i.split("_")[0]) for i in df_merged.columns]
    df_merged.loc["IndexNumber"] = indexnumber
    df_merged = df_merged.sort_values("IndexNumber", axis=1)

    # remove index row
    df_merged = df_merged.iloc[:-1, :]

    return df_merged



def separate_IS(df):
    """
    Separate internal standard and metabolite data into two DataFrames.

    Internal standards are identified by a metabolite ID >= 1000.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing both metabolite and internal standard data

    Returns:
    --------
    tuple
        - df_iso : pd.DataFrame — internal standard data
        - df_metabolites : pd.DataFrame — metabolite data
    """
    metabolite_labels = df.columns
    # get metabolites with ID >= 1000 (corresponds to internal standard IDs)
    ls_isotope = []
    for i in metabolite_labels:
        metabolite_ID = i.split("_")[0]
        if int(metabolite_ID) >= 1000:
            ls_isotope.append(i)

    df_iso = df[ls_isotope]
    df_metabolites = df.drop(columns=ls_isotope)

    return (df_iso, df_metabolites)




def analyze_cohort_distribution(sample_ids):
    """
    Analyze the distribution of cohorts across batches.
    
    Parameters:
    -----------
    sample_ids : list
        List of sample IDs
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with batch statistics
    """
    # Set the seaborn theme
    sns.set_theme(style='white')
    
    # Create a DataFrame from the sample IDs
    df = pd.DataFrame({'sample_id': sample_ids})
    
    # Identify cohort based on the first character of sample ID
    def determine_cohort(sample_id):
        if sample_id.startswith('1'):
            return 'Cohort 1'
        elif sample_id.startswith('2') or sample_id.startswith('3'):
            return 'Cohort 2'
        elif sample_id.startswith('M4D'):
            return 'QC'
        else:
            return 'Unknown'
    
    # Extract batch number from sample ID
    def extract_batch(sample_id):
        parts = sample_id.split('_')
        if len(parts) > 2:
            return parts[-2]
        else:
            return 'Unknown'
    
    # Apply the functions to create new columns
    df['cohort'] = df['sample_id'].apply(determine_cohort)
    df['Batch'] = df['sample_id'].apply(extract_batch)
    
    # Filter out QC samples, focus on Cohorts 1 and 2
    analysis_df = df[df['cohort'].isin(['Cohort 1', 'Cohort 2'])]
    
    # Count samples by batch and cohort
    batch_cohort_counts = pd.crosstab(analysis_df['Batch'], analysis_df['cohort'])
    
    # Ensure both cohorts are present in the counts
    for cohort in ['Cohort 1', 'Cohort 2']:
        if cohort not in batch_cohort_counts.columns:
            batch_cohort_counts[cohort] = 0
    
    # Calculate total samples and proportion of Cohort 2
    batch_cohort_counts['Total'] = batch_cohort_counts.sum(axis=1)
    batch_cohort_counts['Cohort_2_Proportion'] = batch_cohort_counts['Cohort 2'] / batch_cohort_counts['Total']
    
    # Create a tidy version of the data for plotting
    plot_data = batch_cohort_counts.reset_index().rename(columns={'index': 'Batch'})
    
    # Try to sort batches numerically if possible
    try:
        plot_data['Batch_Num'] = plot_data['Batch'].str.extract(r'(\d+)', expand=False).astype(float)
        plot_data = plot_data.sort_values('Batch_Num')
        plot_data = plot_data.drop('Batch_Num', axis=1)
    except:
        # If sorting fails, keep original order
        pass
    
    return plot_data



def plot_cohort2_proportion(data):
    """
    Create a plot showing the proportion of Cohort 2 samples by batch.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with batch statistics

    Returns:
    --------
    matplotlib.axes.Axes
        The generated axes object
    """
    plt.figure(figsize=(10, 6))
    
    # Create the bar plot
    ax = sns.barplot(x='Batch', y='Cohort_2_Proportion', data=data)
      
    # Add titles and labels
    plt.title('Proportion of Patient Samples by Batch')
    plt.xlabel('Batch')
    plt.ylabel('Proportion of Patient Samples')
    
    # Rotate x-axis labels if there are many batches
    if len(data) > 5:
        plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.show()
    
    return ax



def fillna_replicates(df_merged):
    """
    Fill missing values with the mean of available replicates.

    Values stay NaN if all replicates are missing for that feature.

    Parameters:
    -----------
    df_merged : pd.DataFrame
        DataFrame with merged positive and negative mode data

    Returns:
    --------
    pd.DataFrame
        DataFrame with NaN values filled where replicate means are available
    """
    # for each sample
    for sample in df_merged.index:

        # get replicates
        i = sample.split("_")[0]
        df_rep = df_merged.loc[df_merged.index.str.contains(i)]

        # get na indices
        na_id = np.argwhere((df_rep.loc[sample].isnull()))

        # fillna replicate mean
        for j in na_id:
            mean = df_rep.iloc[:, j[0]].mean()
            # if mean of replicates is nan -> ignore
            if np.isnan(mean):
                continue
            # replace nan value with mean
            df_merged.loc[sample, df_rep.columns[j[0]]] = mean

    return df_merged



def separate_cohorts(df_merged):
    """
    Separate data into DataFrames for each cohort and QC samples.

    Cohort assignment is based on the first character(s) of the sample ID:
    '1' → Cohort 1, '2' → Cohort 2, '3' → Cohort 3, 'M4D' → QC.

    Parameters:
    -----------
    df_merged : pd.DataFrame
        DataFrame containing merged positive and negative mode data

    Returns:
    --------
    list of pd.DataFrame
        [df_cohort_1, df_cohort_2, df_cohort_3, df_QC]
    """
    sample_IDs = df_merged.index
    IDs_cohort_1 = []
    IDs_cohort_2 = []
    IDs_cohort_3 = []
    IDs_QC = []

    for i in sample_IDs:
        if i.startswith("1"):
            IDs_cohort_1.append(i)
        elif i.startswith("2"):
            IDs_cohort_2.append(i)
        elif i.startswith("3"):
            IDs_cohort_3.append(i)
        elif i.startswith("M4D"):
            IDs_QC.append(i)

    IDs_cohort_2 = sorted(set(IDs_cohort_2), key=IDs_cohort_2.index)
    IDs_cohort_3 = sorted(set(IDs_cohort_3), key=IDs_cohort_3.index)

    df_cohort_1 = df_merged.loc[IDs_cohort_1]
    df_cohort_2 = df_merged.loc[IDs_cohort_2]
    df_cohort_3 = df_merged.loc[IDs_cohort_3]
    df_QC = df_merged.loc[IDs_QC]

    return [df_cohort_1, df_cohort_2, df_cohort_3, df_QC]



def age_filter(df, ages):
    """
    Filter samples by age group identifiers.

    Parameters:
    -----------
    df : pd.DataFrame
        Samples DataFrame
    ages : list of str
        Age group identifiers to retain

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only samples matching the specified age groups
    """
    df_list = []
    for i in ages:
        sub_df = df.loc[df.index.str.contains(i + "_")]
        df_list.append(sub_df)
    df_filtered = pd.concat(df_list)

    return df_filtered



def log_metabolite_ratios(df):
    """
    Calculate and append log-space metabolite ratios to the DataFrame.

    Adds the following ratio columns: Spermidine/Spermine, Gln/Glu, Gln/Urea,
    (Gln+Glu)/Urea, Creatine/Creatinine, and Pyridoxamine/4-Pyridoxate.
    Ratios whose input features are not in the panel are skipped.

    Parameters:
    -----------
    df : pd.DataFrame
        Log-transformed metabolite data

    Returns:
    --------
    pd.DataFrame
        Input DataFrame with ratio columns appended
    """
    # check if all features needed for a ratio are in the panel
    def has_features(*features):
        missing = [f for f in features if f not in df.columns]
        if missing:
            print('Ratio skipped, feature(s) not found: ' + ', '.join(missing))
        return not missing

    # Spermidine/Spermine
    if has_features("229_Spermidine_pos", "228_Spermine_pos"):
        ssratio = df["229_Spermidine_pos"] - df["228_Spermine_pos"]
        idx = len(df.columns) - 1
        df.insert(loc=idx, column="900_Spermidine-Spermine_ratio", value=ssratio)

    # Gln/Glu
    if has_features("095_Glutamine_pos", "096_L-GlutamicAcid_pos"):
        ssratio1 = df["095_Glutamine_pos"] - df["096_L-GlutamicAcid_pos"]
        idx = len(df.columns) - 1
        df.insert(loc=idx, column="901_Gln-Glu_ratio", value=ssratio1)

    # Gln/Urea
    if has_features("095_Glutamine_pos", "255_Urea_pos"):
        ssratio2 = df["095_Glutamine_pos"] - df["255_Urea_pos"]
        idx = len(df.columns) - 1
        df.insert(loc=idx, column="902_Gln-Urea_ratio", value=ssratio2)

    # (Gln+Glu)/Urea
    if has_features("095_Glutamine_pos", "096_L-GlutamicAcid_pos", "255_Urea_pos"):
        df["903_Gln-Glu-Urea_ratio"] = np.log2((
            np.exp2(df["095_Glutamine_pos"]) + np.exp2(df["096_L-GlutamicAcid_pos"])
        ) / np.exp2(df["255_Urea_pos"]))

    # Creatine/Creatinine
    if has_features("065_Creatine_pos", "066_Creatinine_pos"):
        ssratio4 = df["065_Creatine_pos"] - df["066_Creatinine_pos"]
        idx = len(df.columns) - 1
        df.insert(loc=idx, column="904_Creatine_Creatinine_ratio", value=ssratio4)

    # Pyridoxamine/4-Pyridoxic acid
    if has_features("172_Pyridoxamine_pos", "173_4Pyridoxate_pos"):
        ssratio5 = df["172_Pyridoxamine_pos"] - df["173_4Pyridoxate_pos"]
        idx = len(df.columns) - 1
        df.insert(loc=idx, column="905_Pyridoxamine_Pyridoxate_ratio", value=ssratio5)

    return df



def log_IS_correction(df, IS):
    """
    Normalize metabolite data using corresponding internal standards in log space.

    Each metabolite is normalized by subtracting the log-transformed IS signal.
    If an IS is not measured, a fallback value of 10 is subtracted.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing both metabolite and IS data
    IS : pd.DataFrame
        DataFrame mapping metabolites to their corresponding internal standards

    Returns:
    --------
    pd.DataFrame
        DataFrame containing IS-normalized metabolite data
    """

    # separate IS and met
    df_IS, df_met = separate_IS(df)

    # scale metabolites by corresponding IS
    for i in df_met.columns:
        if i in IS.index:
            IScol = str(IS.loc[i, "IS"])
            if IScol in df_IS.columns:
                df_met[i] = df_met[i] - df_IS[IScol]
            else:
                print("IS not measured")
                df_met[i] = df_met[i] - 10

    return df_met



def calc_FC(df_controls, df_patients):
    """
    Calculate fold changes for patient samples relative to control means.

    Parameters:
    -----------
    df_controls : pd.DataFrame
        Control cohort data
    df_patients : pd.DataFrame
        Patient cohort data

    Returns:
    --------
    pd.DataFrame
        DataFrame containing fold changes
    """
    c_means = df_controls.mean()
    c_means[-1] = 1
    df_fc = df_patients / c_means
    return df_fc



def encode_labels(cohort_list, include_c3=True):
    """
    Encode class labels for cohorts: 0 for Cohort 1, 1 for Cohorts 2 and 3.

    Parameters:
    -----------
    cohort_list : list of pd.DataFrame
        List of DataFrames for all cohorts, as returned by separate_cohorts
    include_c3 : bool, optional
        Whether to include Cohort 3 in the analysis, by default True

    Returns:
    --------
    pd.DataFrame
        DataFrame with class labels appended
    """
    df_c1 = cohort_list[0]
    df_c2 = cohort_list[1]
    df_c3 = cohort_list[2]

    # encode class labels
    df_c1['Class'] = [0]*len(df_c1)
    df_c2['Class'] = [1]*len(df_c2)

    # decides if cohort 3 is included or not
    if include_c3:
        df_c3['Class'] = [1]*len(df_c3)
        df_encoded = pd.concat([df_c1, df_c2, df_c3])
    else:
        df_encoded = pd.concat([df_c1, df_c2])

    return(df_encoded)
