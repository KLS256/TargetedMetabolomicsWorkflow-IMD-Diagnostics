# Imports

# 1. Standard library
import os
import time

# 2. Third-party
import numpy as np
import pandas as pd


# 3. Own modules
from . import batch_normalization_modules as bnm
from . import gmm_analysis_modules as gam
from . import nan_analysis_modules as nam
from . import outlier_modules as om
from . import plotting_modules as plm
from . import scaling_modules as sm
from . import stats_modules as stm
from . import targeted_workflow_modules as tw


# Functions

def read_and_filter_data(input_path, output_path, out_intermediate, sampletype, remove, params, 
                         normalization_method, check_stability, batch_normalization):
    """
    Read targeted MS data, apply quality filters, and optionally apply batch normalization.

    Writes processed data to the output folder. 
    Parameters:
    -----------
    input_path : str
        Directory containing raw data
    output_path : str
        Root output directory
    out_intermediate : str
        Directory for intermediate output files
    sampletype : str
        Sample type: 'U' for urine, 'P' for plasma
    remove : list
        Batch names to exclude from analysis
    params : dict
        User-defined processing parameters
    normalization_method : str
        Batch normalization method: 'QC' or 'Samples'
    check_stability : bool
        Whether to apply stability checks during batch normalization
    batch_normalization : bool
        Whether to apply batch normalization

    Returns:
    --------
    tuple
        - output_path : str — sampletype-specific output subdirectory
        - day_path : str — path to the dated run folder
    """
    start = time.time()

    # Write parameters in run-folder
    df_params = pd.DataFrame.from_dict(params, orient='index')
    df_params.to_csv(output_path + "/parameters.csv")

    # Print Sampletype
    print('Sampletype: ' + sampletype)
     
    # Keep output path for nan-plots
    out_dir = output_path
    # Generate subdirectories
    if sampletype == "P":
        output_path = output_path + "/Plasma/"
    elif sampletype == "U":
        output_path = output_path + "/Urine/"
    
    # Save Path to day folder for loading saved df_wnan
    day_path = os.path.dirname(out_dir)


    # Load & filter Data 
    print('Loading Data.')

    # Generate Input Folder Lists
    dirs = tw.input_directories(input_path, sampletype)

    # Remove outlier batches
    dirs = [x for x in dirs if x not in remove]

    # Read target labels
    metabloc = input_path + "metabolites.csv"
    df_metabolites = pd.read_csv(metabloc, index_col=0)

    # Read outlier-filtered data
    df_samples, batchlist = tw.read_MS_data(input_path, dirs, df_metabolites)

    # Generate Sample-Batch map
    df_batches = nam.sample_batch_map(df_samples, batchlist)

    # Drop blanks
    df_samples = tw.drop_blanks(df_samples)
    df_batches = tw.drop_blanks(df_batches.T)

    # Transpose sample matrix and fix labels
    df_samples = tw.transpose_samples(df_samples)
    df_batches = df_batches.T

    # Append batch ids to sample ids
    df_samples = tw.append_batch_to_id(df_samples, df_batches)

    # Merge pos and neg modes
    df_merged = tw.merge_modes(df_samples)
    # Drop empty samples (corrupt)
    df_merged = df_merged.dropna(how='all')

    # NaN analysis based on QC samples
    qc_samples = nam.qc_batch_map(df_merged)
    qc_nan_prevalence = round(qc_samples.isna().sum().sum()/(qc_samples.shape[0]*qc_samples.shape[1]),3)
    print(f'QC NaN Prevalence: {qc_nan_prevalence}')
    qc_batch_nan = nam.qc_nan_per_batch(qc_samples, sampletype, day_path)
    nam.nan_batch_stats(qc_batch_nan)

    # Separate internal standards
    df_IS, df_metabo = tw.separate_IS(df_merged)

    # Plot Patient Proportion per Cohort
    id_list = df_merged.index.tolist()
    batch_composition = tw.analyze_cohort_distribution(id_list)
    pat_proportion_plot = tw.plot_cohort2_proportion(batch_composition)
    pat_proportion_plot.figure.savefig(out_intermediate + '/pat_cohort_proportion_per_batch.png')

    if batch_normalization:
        # Normalize Batches based on QC or all Samples
        df_merged = bnm.batch_normalization(data=df_merged, out_dir=out_intermediate, 
                                            nan_threshold=0.2, normalization_method=normalization_method,
                                            check_stability=check_stability)

    else:
        print('No Batch Normalization!')

    # Visualize Missing Data before Imputation
    nam.visualize_missing_data(df=df_merged, save_path=out_intermediate)

    # Fillna with replicate means
    df_merged = tw.fillna_replicates(df_merged)

    # Clean lysoSM509 label
    labels = list(df_merged.columns)
    labels = ['247_lysoSM509_pos' if x == '247_lysoSM609_pos' else x for x in labels] # fix labelling error in processing of raw data
    df_merged.columns = labels

    # Delete feature 226 because measurements are unreliable (only if present in the panel)
    if '226_OxalaceticAcid_neg' in df_merged.columns:
        del df_merged['226_OxalaceticAcid_neg']

    # Separate the three Cohorts & Control and remove QC samples
    cohort_list = tw.separate_cohorts(df_merged)
    df_ctrl_wnan = cohort_list[0]
    df_pat_wnan = pd.concat([cohort_list[1], cohort_list[2]])

    # Save dataframes to avoid reprocessing
    df_ctrl_wnan.to_csv(day_path + "/df_ctrl_wnan_" + sampletype + ".csv")
    df_pat_wnan.to_csv(day_path + "/df_pat_wnan_" + sampletype + ".csv")
    df_batches.to_csv(day_path + "/df_batches_" + sampletype + ".csv")
    qc_samples.to_csv(day_path+ "/df_qc_" + sampletype + ".csv")

    # Intermediate files (not overwritten)
    df_ctrl_wnan.to_csv(out_intermediate + "/df_ctrl_wnan_" + sampletype + ".csv")
    df_pat_wnan.to_csv(out_intermediate + "/df_pat_wnan_" + sampletype + ".csv")
    df_batches.to_csv(out_intermediate + "/df_batches_" + sampletype + ".csv")
    qc_samples.to_csv(out_intermediate + "/df_qc_" + sampletype + ".csv")


    end = time.time()
    mins = (end-start)/60
    print('Reading & Filtering Data: ' + str(round(mins,2)) + ' Minutes')
    return(output_path, day_path)



def process_data(
    output_path, out_intermediate, sampletype, iqr_cutoff,
    zero_threshold, numneighbors, scaling,
    gmm_model_type, filter_noise, weight_noise_threshold, noise_probability_threshold,
    ages, nan_imputation = 'IEM', outlier_detection = True, mstus = True, 
    IS_correction = True, is_list = None
):
    """
    Remove outlier samples, normalize, scale, and impute missing values in control and patient data.

    Performs log transformation, optional outlier detection, MSTUS normalization,
    metabolite ratio calculation, GMM analysis, scaling, and KNN imputation.
    Writes imputed data to the output folder.

    Parameters:
    -----------
    output_path : str
        Sampletype-specific output directory
    out_intermediate : str
        Directory for intermediate output files
    sampletype : str
        Sample type: 'U' for urine, 'P' for plasma
    iqr_cutoff : float
        IQR multiplier for outlier detection
    zero_threshold : float
        NaN fraction above which features are imputed with half-minimum scatter
    numneighbors : int
        Number of neighbors for KNN imputation
    scaling : str
        Scaling method: 'median', 'mean', 'uv', 'pareto'
    gmm_model_type : str
        GMM type: 'regular', 'bayesian', or 'both'
    filter_noise : bool
        Whether to filter noise components identified by GMM analysis
    weight_noise_threshold : float
        GMM component weight threshold for noise classification
    noise_probability_threshold : float
        Probability threshold for noise filtering
    ages : list of str
        Age group identifiers to include in the analysis
    nan_imputation : str, optional
        Imputation strategy: 'IEM' for two-stage KNN, otherwise half-minimum, by default 'IEM'
    outlier_detection : bool, optional
        Whether to apply IQR and Isolation Forest outlier detection, by default True
    mstus : bool, optional
        Whether to apply MSTUS concentration normalization, by default True
    IS_correction : bool, optional
        Whether to apply internal standard correction, by default True

    Returns:
    --------
    tuple
        - cohort_list : list of pd.DataFrame — DataFrames for each cohort
        - df_ctrl : pd.DataFrame — control cohort data
        - df_pat : pd.DataFrame — patient cohort data
        - ctrl_nan_per_col : pd.Series — NaN fractions per feature in the control cohort
        - results_dict_2 : dict — GMM results from post-processing
    """
    start = time.time()

    print('Processing data.')

    # Initialize dfs
    df_ctrl, df_pat, df_batches = None, None, None  

    # Load df_ctrl and df_pat and generate ctrl_nan_per_col
    # Read Data
    if sampletype == 'P':
        df_ctrl = pd.read_csv(out_intermediate + "/df_ctrl_wnan_P.csv", index_col="000_Sample_number")
        df_pat = pd.read_csv(out_intermediate + "/df_pat_wnan_P.csv", index_col="000_Sample_number")
        df_batches = pd.read_csv(out_intermediate + "/df_batches_P.csv", index_col="Unnamed: 0")

    elif sampletype == 'U':
        df_ctrl = pd.read_csv(out_intermediate + "/df_ctrl_wnan_U.csv", index_col="000_Sample_number")
        df_pat = pd.read_csv(out_intermediate + "/df_pat_wnan_U.csv", index_col="000_Sample_number")
        df_batches = pd.read_csv(out_intermediate + "/df_batches_U.csv", index_col="Unnamed: 0")

    # Age filter here to respect entire processing
    df = pd.concat([df_ctrl, df_pat])
    df = tw.age_filter(df, ages)
    cohort_list = tw.separate_cohorts(df)
    df_ctrl = cohort_list[0]
    df_pat = pd.concat([cohort_list[1], cohort_list[2]])

    nan_col = df_ctrl.isna().sum()
    # get percentage of nan for each column
    ctrl_nan_per_col = nan_col/len(df_ctrl)

    # Log Transformation
    df_ctrl = sm.log2_transformation(df_ctrl)
    df_pat = sm.log2_transformation(df_pat)

    if outlier_detection:
        # Outlier Detection:
        print('Outlier removal.')

        # iForest outlier detection
        # fillna for iForest
        df_ctrl_temp_imp = df_ctrl.fillna(df_ctrl.median())
        # fillna of all na features
        df_ctrl_temp_imp = df_ctrl_temp_imp.fillna(1)
        ano_score, outliers = om.iForest(df_ctrl_temp_imp)
        outliers = pd.DataFrame(outliers, index=df_ctrl.index, columns=["Label"])
        # filter out iForest outliers
        df_ctrl = df_ctrl.loc[outliers["Label"] == 1]
        df_ctrl.to_csv(out_intermediate + '/df_ctrl_wo_outliers_' + sampletype +'.csv')

        # Cohort 1: IQR
        outlier_labels = om.IQR_nan(df_ctrl, cutoff_factor=iqr_cutoff)
        df_ctrl = om.IQR_outlier_removal(df_ctrl, outlier_labels)
        
    # Normalize data before imputation
    df = pd.concat([df_ctrl, df_pat])
    # Visualize Missing Data before Imputation
    nam.visualize_missing_data(df, save_path=out_intermediate)
    df_standard, df_metabo = tw.separate_IS(df)     # Remove IS for normalization
    # Normalize for concentration differences
    if mstus:
        df = sm.log_urine_conc_normalization_nan_threshold(df_metabo, df_standard, nan_threshold=0.2)

    else:
        print('No MSTUS')

    # Add selected metabolite ratios (be careful with log space)
    df = tw.log_metabolite_ratios(df)

    # Show IEM clustering in PCA before Scaling & Imputing NaN
    plm.run_pca_analysis(df, save_path=out_intermediate)

    #  Filter for age-groups
    df = tw.age_filter(df, ages)

    # Split data for GMM Analysis
    cohort_list = tw.separate_cohorts(df)
    df_ctrl = cohort_list[0]
    df_pat = pd.concat([cohort_list[1], cohort_list[2]])

    # GMM analysis of mostly unprocessed data
    start_gmm1 = time.time()
    print('GMM Analysis 1.')

    # Depending on noise filter, the output of GMM analysis is different
    if filter_noise:
        results_dict, summary_df, component_probas, filtered_data, changed_points_df = gam.GMM_analysis(
            data=df_ctrl, model_type=gmm_model_type, out_dir=out_intermediate, timepoint='Early_Processing',
            weight_noise_threshold=weight_noise_threshold, filter_noise=True, 
            noise_probability_threshold=noise_probability_threshold 
        )
        df_ctrl = filtered_data
    else:
        results_dict, summary_df, component_probas = gam.GMM_analysis(
            data=df_ctrl, model_type=gmm_model_type, out_dir=out_intermediate, timepoint='Early_Processing',
            weight_noise_threshold=weight_noise_threshold, filter_noise=False, 
            noise_probability_threshold=noise_probability_threshold     
        )

    end_gmm1 = time.time()
    mins = (end_gmm1-start_gmm1)/60
    print('GMM Analysis 1: ' + str(round(mins,2)) + ' min')

    # Join ctrl and pat for scaling
    df = pd.concat([df_ctrl, df_pat])

    # Scaling
    if scaling == 'median':
        df = sm.log_median_scaling(df)
    elif scaling == 'mean':
        df = sm.log_mean_scaling(df)
    elif scaling == 'uv':
        df = sm.log_uv_scaling(df)
    elif scaling == 'pareto':
        df = sm.log_pareto_scaling(df)
    else:
        print('Choose a valid scaling method.')

    # Separate cohorts
    cohort_list = tw.separate_cohorts(df)
    df_ctrl = cohort_list[0]
    df_pat = pd.concat([cohort_list[1], cohort_list[2]])

        # Write to file
    if sampletype == 'P':
        df.to_csv(output_path + "/df_pre_imputation_P.csv")
        df.to_csv(out_intermediate + "/df_pre_imputation_P.csv")

    elif sampletype == 'U':
        df.to_csv(output_path + "/df_pre_imputation_U.csv")
        df.to_csv(out_intermediate + "/df_pre_imputation_U.csv")


    # NaN Imputation
    print('NaN imputation.')
    if nan_imputation == 'IEM':
        # read ctrl df with imputed values (impute separately)
        df_ctrl_knn = nam.zero_kNN(df_ctrl, zero_threshold, numneighbors)
        # impute replicates separately to avoid high value bleeding
        df_pat_imp = nam.kNN_imputation_separate_pat(df_ctrl_knn, df_pat, zero_threshold, numneighbors)
        df_miss = pd.concat([df_ctrl_knn, df_pat_imp])

    else:
        print('Halfmin Imputation')
        df = pd.concat([df_ctrl, df_pat])
        df_miss = nam.half_minimum_point(df)

    # Write to file
    if sampletype == 'P':
        df_miss.to_csv(output_path + "/df_imputed_P.csv")
        df_miss.to_csv(out_intermediate + "/df_imputed_P.csv")

    elif sampletype == 'U':
        df_miss.to_csv(output_path + "/df_imputed_U.csv")
        df_miss.to_csv(out_intermediate + "/df_imputed_U.csv")


    # Load df_miss
    df = pd.read_csv(out_intermediate + '/df_imputed_' + sampletype + '.csv', index_col="000_Sample_number")

    # IS correction
    if IS_correction:
        print('Internal standard correction.')
        ISloc = is_list
        df_IS = pd.read_csv(ISloc, sep='\t', index_col='Sample_number')
        df = tw.log_IS_correction(df, df_IS)

    else:
        print('No IS correction')
        _, df = tw.separate_IS(df)  

    cohort_list = tw.separate_cohorts(df)
    df_ctrl = cohort_list[0]
    df_pat = pd.concat([cohort_list[1], cohort_list[2]])

    # GMM analysis 2 of mostly processed data
    print('GMM Analysis 2.')
    start_gmm2 = time.time()

    # Noise filter not desired here
    results_dict_2, summary_df_2, component_probas_2 = gam.GMM_analysis(
        data=df_ctrl, model_type=gmm_model_type, out_dir=out_intermediate, timepoint='Post_Processing',
        weight_noise_threshold=weight_noise_threshold, filter_noise=False, 
        noise_probability_threshold=noise_probability_threshold,
        plot=False     
        )
    
    end_gmm2 = time.time()
    mins = (end_gmm2-start_gmm2)/60
    print('GMM Analysis 2: ' + str(round(mins,2)) + " min")

    end = time.time()
    mins = (end-start)/60
    print('Time spent for Processing: ' + str(round(mins,2)) + " min")
  
    return (cohort_list, df_ctrl, df_pat, ctrl_nan_per_col, results_dict_2)



def calculate_fold_change(
    out_intermediate, sampletype, fc_outlier_correction, iqr_cutoff, cohort_list, c3
):
    """
    Calculate fold changes and log2 fold changes for patient samples relative to controls.

    Optionally applies IQR-based outlier correction to both FC and log2FC.

    Parameters:
    -----------
    out_intermediate : str
        Directory for intermediate output files
    sampletype : str
        Sample type: 'U' for urine, 'P' for plasma
    fc_outlier_correction : bool
        Whether to apply IQR outlier correction to fold changes
    iqr_cutoff : float
        IQR multiplier for outlier detection
    cohort_list : list of pd.DataFrame
        List of DataFrames for all cohorts, as returned by separate_cohorts
    c3 : bool
        Whether to include Cohort 3 in the analysis

    Returns:
    --------
    tuple
        - df_ctrl : pd.DataFrame — control data
        - df_pat : pd.DataFrame — patient data
        - df_FC_ctrl : pd.DataFrame — control fold changes
        - df_FC_pat : pd.DataFrame — patient fold changes
        - df_lg_FC_ctrl : pd.DataFrame — control log2 fold changes
        - df_lg_FC_pat : pd.DataFrame — patient log2 fold changes
    """
    start_fc = time.time()
    # Transform data back for IS correction and FC calculation

    df = tw.encode_labels(cohort_list, include_c3=c3)
    classes = df['Class']
    # reverse log-transform for fold-change calculation
    df = df.apply(lambda x: np.exp2(x))
    df['Class'] = classes

    # Split Cohorts
    cohort_list = tw.separate_cohorts(df)
    df_ctrl = cohort_list[0]
    df_pat = pd.concat([cohort_list[1], cohort_list[2]])

    # IQR outlier correction
    if fc_outlier_correction:
        print('FC outlier correction.')
        outlier_labels = om.IQR(df_ctrl, cutoff_factor=iqr_cutoff)
        df_ctrl = om.gaussian_IQR_outlier_correction(
            df_ctrl, outlier_labels, center="mean"
        )

    # Get FC for patient samples (normalize by control)
    print('Calculate fold-changes.')
    df_FC_pat = tw.calc_FC(df_ctrl, df_pat)
    df_FC_ctrl = tw.calc_FC(df_ctrl, df_ctrl)

    # Calculate log2 Fold-Changes
    print('Calculate log2 fold-changes.')
    df_lg_FC_pat = sm.log2_transformation(df_FC_pat)
    df_lg_FC_ctrl = sm.log2_transformation(df_FC_ctrl)

    # Outlier correction of logFC
    if fc_outlier_correction:
        print('log2FC outlier correction.')
        outlier_labels = om.IQR(df_lg_FC_ctrl, iqr_cutoff)
        df_lg_FC_ctrl = om.gaussian_IQR_outlier_correction(
            df_lg_FC_ctrl, outlier_labels, center="mean", absolute=False
        )

    df_FC_pat.to_csv(out_intermediate + '/df_FC_pat_' + sampletype +'.csv')
    df_lg_FC_pat.to_csv(out_intermediate + '/df_lg_FC_pat_' + sampletype +'.csv')
    df_FC_ctrl.to_csv(out_intermediate + '/df_FC_ctrl_' + sampletype +'.csv')
    df_lg_FC_ctrl.to_csv(out_intermediate + '/df_lg_FC_ctrl_' + sampletype + '.csv')

    end_fc = time.time()
    mins = (end_fc-start_fc)/60
    print('FC calculation: ' + str(round(mins,2)) + " min")

    return (df_ctrl, df_pat, df_FC_ctrl, df_FC_pat, df_lg_FC_ctrl, df_lg_FC_pat)



def FC_plots(
    output_path, df_FC_ctrl, df_FC_pat, df_lg_FC_ctrl, df_lg_FC_pat, write, sig_lol
):
    """
    Generate violin plots for fold changes and log2 fold changes of significant metabolites.

    Parameters:
    -----------
    output_path : str
        Output directory path
    df_FC_ctrl : pd.DataFrame
        Fold changes of control samples
    df_FC_pat : pd.DataFrame
        Fold changes of patient samples
    df_lg_FC_ctrl : pd.DataFrame
        Log2 fold changes of control samples
    df_lg_FC_pat : pd.DataFrame
        Log2 fold changes of patient samples
    write : bool
        If True, saves plots to the output directory
    sig_lol : pd.DataFrame
        Boolean DataFrame indicating significant metabolites per patient
    """
    start = time.time()

    print('FC Plots (Significant Features)')
    # get set of patient ids
    pat_id_set = ['_'.join(x.split('_')[:2]) for x in df_FC_pat.index]   
    pat_id_set = list(dict.fromkeys(pat_id_set)) 
    # Append plot labels
    df_FC_pat["Class"] = ["Patient"] * len(df_FC_pat)
    df_FC_ctrl["Class"] = ["Control"] * len(df_FC_ctrl)

    # Generate Violin Plots (FC)
    plm.plot_FC_violin(pat_id_set, df_FC_pat, df_FC_ctrl, output_path, write, sig_lol)

    # Append class labels
    df_lg_FC_pat["Class"] = ["Patient"] * len(df_FC_pat)
    df_lg_FC_ctrl["Class"] = ["Control"] * len(df_FC_ctrl)

    # Generate Violin Plots (log2(FC))
    plm.plot_log_FC_violin(pat_id_set, df_lg_FC_pat, df_lg_FC_ctrl, output_path, write, sig_lol)

    end = time.time()
    mins = end-start
    print('Time spent for FC plotting: ' + str(round(mins,2)) + " min")

    print(' ')



def patient_FC_plots(
    patient_list, output_path, df_FC_ctrl, df_FC_pat, df_lg_FC_ctrl, df_lg_FC_pat
):
    """
    Generate violin plots for fold changes of a selected subset of patients.

    All metabolites are plotted regardless of significance.

    Parameters:
    -----------
    patient_list : list of str
        Patient IDs to generate plots for
    output_path : str
        Output directory path
    df_FC_ctrl : pd.DataFrame
        Fold changes of control samples
    df_FC_pat : pd.DataFrame
        Fold changes of patient samples
    df_lg_FC_ctrl : pd.DataFrame
        Log2 fold changes of control samples
    df_lg_FC_pat : pd.DataFrame
        Log2 fold changes of patient samples
    """

    print('FC Plots (Selected Patients)')
    print(patient_list)
    # get set of patient ids
    pat_id_set = list(patient_list)
    
    # Force writing plots for all metabolites
    write = True
    sig_lol = pd.DataFrame(index=pat_id_set, columns=df_FC_pat.columns)
    sig_lol.loc[:,:] = True
    # Append plot labels
    df_FC_pat["Class"] = ["Patient"] * len(df_FC_pat)
    df_FC_ctrl["Class"] = ["Control"] * len(df_FC_ctrl)

    # Generate Violin Plots (FC)
    plm.plot_FC_violin(pat_id_set, df_FC_pat, df_FC_ctrl, output_path, write, sig_lol)

    # Append class labels
    df_lg_FC_pat["Class"] = ["Patient"] * len(df_FC_pat)
    df_lg_FC_ctrl["Class"] = ["Control"] * len(df_FC_ctrl)

    # Generate Violin Plots (log2(FC))
    plm.plot_log_FC_violin(pat_id_set, df_lg_FC_pat, df_lg_FC_ctrl, output_path, write, sig_lol)

    print(' ')



def patient_statistics(output_path, df_ctrl, df_pat, df_FC_pat, mode, p_fxn, results_dict_2, gmm_pvals, mt_correction, ctrl_nan, nan_filter, sig_level, fc_crop, model_type):
    """
    Calculate patient statistics and generate result overview plots.

    Computes basic descriptive statistics, GMM-based p-values or permutation
    p-values, mode probabilities, and writes per-patient Stats.csv files
    and result overview scatter plots.

    Parameters:
    -----------
    output_path : str
        Output directory path
    df_ctrl : pd.DataFrame
        Control cohort data
    df_pat : pd.DataFrame
        Patient cohort data
    df_FC_pat : pd.DataFrame
        Fold changes of patient samples
    mode : str
        Replicate summary statistic: 'mean', 'median', or 'average'
    p_fxn : str
        P-value estimation method: 'permutation'
    results_dict_2 : dict
        GMM results from post-processing (GMM analysis 2)
    gmm_pvals : bool
        If True, uses GMM-based p-values; if False, uses permutation test
    mt_correction : str
        Multiple testing correction method: 'Bonferroni', 'FDR', or ''
    ctrl_nan : pd.Series
        NaN fractions per feature in the control cohort
    nan_filter : float
        Maximum allowed NaN fraction for feature inclusion in output
    sig_level : float
        P-value threshold for significance
    fc_crop : float
        Lower bound for log2 fold change cropping in plots
    model_type : str
        GMM type used for mode probability calculation: 'regular' or 'bayesian'

    Returns:
    --------
    pd.DataFrame
        Boolean DataFrame indicating significant features per patient
    """
    start = time.time()

    print('Calculating statistics.')
    
    # Remove class labels and log-transform back
    dfp = df_pat.copy()
    del dfp['Class']
    dfp = sm.log2_transformation(dfp)
    dfc = df_ctrl.copy()
    del dfc['Class']
    dfc = sm.log2_transformation(dfc)
    dfc = stm.replicate_mean(dfc)                    # one value per control (permutation test, z-scores)


    # Calculate basic stats
    dfp_means, dfp_medians, dfp_var, dfp_std, df_z = stm.basic_stats(dfc, dfp)

    # Calculate significance
    df_significance = gam.calculate_gmm_pvalues_for_triplicates(results_dict_2, dfp_means, model_type=model_type)

    # Multiple testing correction (must happen before mode_probabilities, which copies the p-values)
    df_significance = stm.correct_gmm_pvalues(df_significance, mt_correction)

    # Calculate component probabilities for patient samples
    mode_probs_2, noise_candidates, multi_peak_candidates = gam.mode_probabilities(results_dict_2,
                                                                                    df_significance,
                                                                                    dfp_means, model_type,
                                                                                    probability_threshold=0.05, significance_threshold=1
                                                                                    )


  
    # Estimate p-values with GMMs
    if gmm_pvals:
            # Extract p-values
            # Generate stats file
        df_pval = df_significance.loc[:, df_significance.columns.str.contains('pvalue', case=False)]
        print(df_pval.shape)
        df_pval_old, df_p_inv, df_p_val_size, df_p_zero = stm.stats_pvals_gmm(df_pval, dfp_means)

        sig_lol = stm.write_stats(df_FC_pat, mode_probs_2, df_p_inv, df_p_val_size, df_p_zero, df_z, sig_level, output_path, ctrl_nan, nan_filter, fc_crop)

    # Estimate p-values with permutation test
    else:
        print('Permutation pvals')
        df_pval, df_p_inv, df_p_val_size, df_p_zero = stm.calculate_pvals(dfc, dfp_means, dfp_medians, mode, p_fxn, mt_correction)
        sig_lol = stm.write_stats(df_FC_pat, mode_probs_2, df_p_inv, df_p_val_size, df_p_zero, df_z, sig_level, output_path, ctrl_nan, nan_filter, fc_crop, df_pval=df_pval)

    end = time.time()
    mins = (end-start)/60
    print('Statistical Analysis: ' + str(round(mins,2)) + " min")
    return(sig_lol)

