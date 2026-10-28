#%% Imports

# 1. Standard library
import time

# 2. Third-party
import config

# 3. Own modules
from modules import meta_workflow_modules as mwm
from modules import patient_modules as pam
from modules import targeted_workflow_modules as tw


# Input parameters        
input_path   = str(config.INPUT_PATH)
output_path  = str(config.OUTPUT_PATH)
is_list = str(config.IS_LIST)

sampletypes  = config.SAMPLETYPES


remove = config.REMOVE
c3     = config.C3
ages   = config.AGES

iqr_cutoff           = config.IQR_CUTOFF
fc_outlier_correction = config.FC_OUTLIER_CORRECTION

filter_noise                = config.FILTER_NOISE
w_noise_threshold           = config.W_NOISE_THRESHOLD
noise_probability_threshold = config.NOISE_PROBABILITY_THRESHOLD
GMM_model_type              = config.GMM_MODEL_TYPE
model_type_stats            = config.MODEL_TYPE_STATS

batch_normalization        = config.BATCH_NORMALIZATION
batch_normalization_method = config.BATCH_NORMALIZATION_METHOD
check_stability            = config.CHECK_STABILITY
scaling                    = config.SCALING

nan_imputation = config.NAN_IMPUTATION
zero_threshold = config.ZERO_THRESHOLD
nan_filter     = config.NAN_FILTER
num_neighbors  = config.NUM_NEIGHBORS

mode          = config.MODE
p_fxn         = config.P_FXN
mt_correction = config.MT_CORRECTION
gmm_pvals     = config.GMM_PVALS
sig_level     = config.SIG_LEVEL
fc_crop       = config.FC_CROP

write         = config.WRITE
plot_patients = config.PLOT_PATIENTS

outlier_detection = config.OUTLIER_DETECTION
mstus             = config.MSTUS
IS_correction     = config.IS_CORRECTION

# Gather parameters in dictionary: 
params = {'outlier_detection': outlier_detection, 'MSTUS': mstus,
          'input_path': input_path, 'output_path': output_path,
          'sampletype': '-'.join(sampletypes),
          'removed_batches': remove, 'cohort_3_included': c3, 'ages': ages, 'batch_normalization': batch_normalization,
          'batch_normalization_method': batch_normalization_method, 'check_stability': check_stability,
          'IQR_cutoff': iqr_cutoff, 
          'fc_outlier_correction': fc_outlier_correction,
          'GMM_model_type': GMM_model_type,'filter_noise': filter_noise, 'GMM_noise_threshold': w_noise_threshold,
          'noise_probability_threshold': noise_probability_threshold,
          'scaling': scaling, 'nan_imputation': nan_imputation,
          'zero_threshold': zero_threshold, 'nan_filter': nan_filter,
          'num_neighbors': num_neighbors,
          'empirical-pval-mode': mode, 'p_fxn': p_fxn, 'mt_correction': mt_correction, 'GMM_pvals': gmm_pvals,
          'sign_threshold_plotting': sig_level, 'fc_crop': fc_crop,
          'write_violin': write, 'plot_patients': plot_patients, 'is_list': is_list, 
          }



# Run Workflow

# Time the workflow
start_wf = time.time()

# Generate output directories and set output_path
out_dir = tw.generate_outdirs(output_path)

# Iterate workflow over sampletypes
for sampletype in sampletypes:
    out_intermediate = out_dir +"/Intermediate_Files_" + sampletype

    # Read data, filter outlier batches and fillna replicates
    out_path, day_path = mwm.read_and_filter_data(input_path, out_dir,out_intermediate, sampletype, remove, params,
                                                  batch_normalization_method, check_stability, batch_normalization)

    # Filter outlier samples of control, fillna, GMM analysis
    cohort_list, df_ctrl, df_pat, ctrl_nan, results_gmm_2 = mwm.process_data(day_path, out_intermediate, sampletype, iqr_cutoff,
                                                zero_threshold,
                                                numneighbors=num_neighbors, scaling=scaling, 
                                                gmm_model_type=GMM_model_type, filter_noise=filter_noise, weight_noise_threshold=w_noise_threshold,
                                                noise_probability_threshold=noise_probability_threshold, ages=ages, nan_imputation=nan_imputation,
                                                outlier_detection = outlier_detection, mstus = mstus, IS_correction = IS_correction, is_list = is_list)

    # Calculate fold-changes
    df_ctrl, df_pat, df_FC_ctrl, df_FC_pat, df_lg_FC_ctrl, df_lg_FC_pat = mwm.calculate_fold_change(
            out_intermediate, sampletype, fc_outlier_correction, iqr_cutoff, cohort_list, c3)

    # Calculate common statistics and generate statplots
    sig_lol = mwm.patient_statistics(out_path, df_ctrl, df_pat, df_FC_pat, mode, p_fxn,
                                     results_gmm_2, gmm_pvals, mt_correction, ctrl_nan, nan_filter,
                                     sig_level, fc_crop, model_type_stats)

    # Generate Violin plots of fold-changes of each metabolite
    # All Patients
    mwm.FC_plots(out_path, df_FC_ctrl, df_FC_pat, df_lg_FC_ctrl, df_lg_FC_pat, write, sig_lol)

    # Specific Patients
    patient_list = [i for i in plot_patients if i[1]==sampletype]
    if len(patient_list) > 0:
        mwm.patient_FC_plots(patient_list, out_path, df_FC_ctrl, df_FC_pat, df_lg_FC_ctrl, df_lg_FC_pat)

# Gather results in Patient folders
pam.gather_patient_data(sampletypes, results_path=out_dir)

# Evaluate runtime of workflow
end_wf = time.time()
print('Completion of the entire workflow: ' + str(round((end_wf-start_wf)/60,2)) + ' Minutes.')


# %%
