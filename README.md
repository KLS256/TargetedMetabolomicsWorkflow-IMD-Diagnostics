# TargetedMetabolomicsWorkflow-IMD-Diagnostics

Supplementary code for:

> Stocker KL, et al. Improving the detection of inherited metabolic disorders by leveraging control cohorts in metabolomics to overcome sample size limitations. Clin Chem Lab Med. 2026. https://doi.org/10.1515/cclm-2026-0794

---

## Code authors

- Kim L. Stocker (primary)
- Martin Poms

---

## Overview

This workflow implements a targeted LC-MS metabolomics pipeline for the diagnosis of inherited metabolic disorders (IMDs). It processes raw per-sample intensity files through batch normalization, outlier removal, missing value imputation, Gaussian mixture model (GMM) fitting, and empirical p-value calculation, and produces per-patient statistical result files and plots.

---

## Installation

Developed and tested with Python 3.10.12

**1. Clone the repository:**

```bash
git clone https://github.com/KLS256/TargetedMetabolomicsWorkflow-IMD-Diagnostics.git
cd TargetedMetabolomicsWorkflow-IMD-Diagnostics
```
> `git clone` automatically creates a new folder named after the repository
> in your current directory. Navigate to the location where you want this
> folder to be created before running the command.

**2. Create and activate a virtual environment:**

Linux / macOS:

```bash
python3.10 -m venv venv
source venv/bin/activate
```

Windows:

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

To leave the virtual environment when you're done, run `deactivate`.

> **Note:** `interactive_statsplot.py` requires a GUI matplotlib backend (e.g. Qt or Tk) and a display, because it uses interactive hover via `mplcursors`. On a headless machine (e.g. a remote server with no display) the interactive plot will not work.

Requirements:
```
numpy==1.26.4
pandas==2.2.2
scipy==1.11.4
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.2
mplcursors==0.6
distfit==1.8.0
PyQt5==5.15.10
```

---

## File structure

```
├── config.py
├── targeted_workflow_final.py
├── interactive_statsplot.py
├── requirements.txt
└── modules/
    ├── __init__.py
    ├── meta_workflow_modules.py
    ├── batch_normalization_modules.py
    ├── gmm_analysis_modules.py
    ├── interactive_statsplot_modules.py
    ├── nan_analysis_modules.py
    ├── outlier_modules.py
    ├── patient_modules.py
    ├── plotting_modules.py
    ├── scaling_modules.py
    ├── stats_modules.py
    └── targeted_workflow_modules.py
```

---

## Input data

The workflow expects the following files in `INPUT_PATH` (set in `config.py`):

- Tab-separated files with sample intensity measurements, organised into one subfolder per batch
  (batch folder names must contain `U` for urine or `P` for plasma)
- `metabolites.csv` — feature labels and metadata (comma-separated)
- An internal standards list (tab-separated, columns `Sample_number` and `IS`; path set via `IS_LIST` in `config.py`)

All folder paths in `config.py` must end with `/`.

---

## How to run

**1. Edit `config.py`** to set your input/output paths and processing parameters. All user-facing settings are documented inline in that file.

**2. Run the main workflow:**

```bash
python targeted_workflow_final.py
```

Results are written to `OUTPUT_PATH` as defined in `config.py`. Each run creates a timestamped subfolder containing:

- `parameters.csv` — full record of all parameters used
- `Intermediate_Files_[U/P]/` — intermediate DataFrames at each processing step
- `Patients/` — per-patient subfolders, each containing a `Stats.csv` with per-feature statistics

**3. Inspect individual patient results interactively:**

Edit `interactive_statsplot.py` to set the patient ID and run folder, then:

```bash
python interactive_statsplot.py
```

This opens an interactive scatter plot of per-feature statistics for the specified patient. Hovering over a point displays the feature name.

---

## Citation

If you use this code, please cite the paper above.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
