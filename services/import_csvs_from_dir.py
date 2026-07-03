
import os
import pandas as pd

# printing
from services.printing.print_warning import print_warning

# DEPRICATED: replaced by get_dfs
def import_csvs_from_dir(dict_datasets, dir_datasets) -> dict:

    # iterate over all files in the folder
    for file_name in os.listdir(dir_datasets):
        
        # check if the file is a CSV file
        if file_name.endswith(".csv"):
            
            # read the CSV file into a dataframe
            file_path = os.path.join(dir_datasets, file_name)
            df = pd.read_csv(file_path)
            
            # take csv from filename
            name_df = os.path.splitext(file_name)[0]

            # add dataset to list
            dict_datasets[name_df] = df
            
    return dict_datasets
