# Imports

# 1. Standard library
import sys
from pathlib import Path

# 2. Third-party
import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd

# 3. Own modules (sys.path must be set before local imports)
sys.path.append(str(Path(__file__).parent / 'modules'))
import interactive_statsplot_modules as ism
import config

# Enter Patient ID and output path for the run you want to inspect
pat_id = "YOUR_SAMPLE_ID"   

# Set the run subfolder (date and run number are created by the workflow)
location = str(config.OUTPUT_PATH + "YYYY-MM-DD/" + "Run1") + "/"

path = location + 'Patients/' + pat_id[2:5] + '/' + pat_id + '/Stats.csv'
df_stats = pd.read_csv(path, index_col=0)

ism.create_interactive_statsplot(df_stats, show_plot=True, title=pat_id)