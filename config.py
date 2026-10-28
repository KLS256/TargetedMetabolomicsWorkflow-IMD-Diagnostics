"""
User configuration for TargetedMetabolomicsWorkflow.
Edit this file before running the workflow.
"""



# =============================================================================
# Paths  —  set these to match your local directory structure
# All folder paths must end with "/"
# =============================================================================

INPUT_PATH = "/path/to/your/input/data/"
# Folder containing one subfolder per batch (tab-separated sample files) and metabolites.csv

OUTPUT_PATH = "/path/to/your/output/"
# Folder where results will be written (created automatically if it does not exist)

IS_LIST = INPUT_PATH + "IST_list.csv"
# Path to internal standards list (tab-separated, columns "Sample_number" and "IS")

# =============================================================================
# Sample configuration
# =============================================================================

SAMPLETYPES = ["U","P"]
# Which sample types to process: "U" (urine), "P" (plasma), or both ["U", "P"]

REMOVE = []
# Batch IDs to exclude from the analysis, e.g. ["BATCH_001"]
# Leave empty to include all samples.

C3 = True
# Include Cohort 3 in the analysis (True / False)

AGES = ["A", "B", "C", "D", "E", "F", "G"]
# Age groups to include

PLOT_PATIENTS = []
# Sample IDs incl. batch (as in the output folder names), for individual FC-plots
# of all metabolites
# Leave empty to skip individual patient plots.


# =============================================================================
# Outlier processing
# =============================================================================

IQR_CUTOFF = 3
# IQR multiplier for outlier removal in the control cohort

FC_OUTLIER_CORRECTION = False
# Apply outlier correction after fold-change calculation (True / False)
# Experimental: developed during method development, not used for published results


# =============================================================================
# GMM analysis
# =============================================================================

FILTER_NOISE = False
# Remove feature values assigned to the noise component (True / False)

W_NOISE_THRESHOLD = 0.4
# Maximum mixture weight allowed for the noise component

NOISE_PROBABILITY_THRESHOLD = 0.9
# Probability threshold for assigning a data point to the noise component

GMM_MODEL_TYPE = "bayesian"
# GMM variant to fit: "regular", "bayesian", or "both"

MODEL_TYPE_STATS = "bayesian"
# GMM variant used for statistics: "regular" or "bayesian"


# =============================================================================
# Data scaling
# =============================================================================

BATCH_NORMALIZATION = True
BATCH_NORMALIZATION_METHOD = "Samples"
# Batch normalisation reference: "Samples" or "QC"

CHECK_STABILITY = False
# Run stability check for batch normalisation (True / False)

SCALING = "mean"
# Scaling method: "mean", "median", "uv", "pareto"


# =============================================================================
# NaN imputation
# =============================================================================

NAN_IMPUTATION = "IEM"
# Imputation strategy: "IEM" uses the IEM-aware strategy; any other string skips it

ZERO_THRESHOLD = 0.6
# Features with a NaN fraction above this value are filled with half the minimum

NAN_FILTER = 0.6
# NaN threshold for feature exclusion in statistics / statplots
# (features above this threshold are excluded unless patient-significant)

NUM_NEIGHBORS = 5
# Number of neighbours for kNN imputation


# =============================================================================
# Statistics
# =============================================================================

MODE = "mean"
# Reference value for empirical p-value calculation: "mean", "median", or "average"

P_FXN = "permutation"
# P-value method (only used if GMM_PVALS = False): "permutation"

MT_CORRECTION = "FDR"
# Multiple-testing correction: "bonferroni", "FDR", or "" (no correction)

GMM_PVALS = True
# Use GMM fit for p-values (True) or use P_FXN method (False)

SIG_LEVEL = 1
# Significance threshold for keeping features above NAN_FILTER (1 = keep all features, i.e. filter disabled)

FC_CROP = -6
# Lower bound for fold-change axis in plots


# =============================================================================
# Plotting
# =============================================================================

WRITE = False
# Write violin plots for all metabolites (True / False)

OUTLIER_DETECTION = True
MSTUS = True
IS_CORRECTION = True