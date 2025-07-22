import pandas as pd
import numpy as np 
from data_handling import *
import os

# Create some variables
results = pd.DataFrame()

# Select .h5 files
filenames = select_file(multiple=True)

# Extract data and concatenate in "results"
for filename in filenames:
    barnes_info = pd.read_hdf(filename)
    results = results.append(barnes_info, ignore_index=True) 
    
# Save file in xlsx
# save_filename = os.path.dirname(filename)+'/'+'Final_results'+'.xlsx'
#save_filename = os.path.dirname(filename)+'/'+'Final_results_probe'+'.xlsx'
#save_filename = os.path.dirname(filename)+'/'+'Final_results_probe_holes'+'.xlsx'
save_filename = os.path.dirname(filename)+'/'+'Final_results_probe_30sec_bins'+'.xlsx'
results.to_excel(save_filename)  