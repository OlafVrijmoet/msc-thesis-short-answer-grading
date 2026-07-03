

import os
import pandas as pd

# services
from services.get_df import get_df

def get_dfs(dict: dict, dir: str):

    # iterate over all files in the folder
    for file_name in os.listdir(dir):

        # get file name without file type for get_df
        file_name, _ = os.path.splitext(file_name)

        # get the df
        found, df_name, df = get_df(dir, file_name)

        if found == True:

            # add df to given dict
            dict[df_name] = df

    # return the given dict with found dfs added
    return dict
