
import os
import pandas as pd
import torch

def get_df(dir, file_name, know_type=None): # can get rid of parquet

    if know_type != None:

        if os.path.exists(f"{dir}/{file_name}.parquet") and know_type == "parquet":

            return get_parquet(dir, file_name)

        elif os.path.exists(f"{dir}/{file_name}.csv") and know_type == "csv":

            # return info on existance
            return get_csv(dir, file_name)

        elif os.path.exists(f"{dir}/{file_name}.pth") and know_type == "pth":

            return get_pth(dir, file_name)
        
        else:

            print(f"File {file_name} not found in ({dir}) with type {know_type}")

            # return info on existance
            return False, file_name, None
    
    # check if parquet file exsits
    if os.path.exists(f"{dir}/{file_name}.parquet"):

        return get_parquet(dir, file_name)
    
    elif os.path.exists(f"{dir}/{file_name}.csv"):

        # return info on existance
        return get_csv(dir, file_name)
    
    elif os.path.exists(f"{dir}/{file_name}.pth"):

        return get_pth(dir, file_name)
    
    else:

        print(f"File {file_name} not found in ({dir})")

        # return info on existance
        return False, file_name, None


def get_parquet(dir, file_name):

    df = pd.read_parquet(f"{dir}/{file_name}.parquet")

    print(f"Found {file_name} with type parquet in ({dir}) and converted it to df")

    # return info on existance
    return True, file_name, df

def get_csv(dir, file_name):

    df = pd.read_csv(f"{dir}/{file_name}.csv")

    print(f"Found {file_name} with type csv in ({dir}) and converted it to df")
    
    # return info on existance
    return True, file_name, df

def get_pth(dir, file_name):

    df = torch.load(f'{dir}/{file_name}.pth')

    print(f"Found {file_name} with type csv in ({dir}) and converted it to df")

    # return info on existance
    return True, file_name, df
