
import os
import torch

def save(dir, file_name, df, parquet=False, file_type=None):

    if not os.path.exists(dir):
        os.makedirs(dir)

    if file_type != None:

        if file_type == "parquet":
            df.to_parquet(f"{dir}/{file_name}.parquet", index=False)

        elif file_type == "csv":
            df.to_csv(f"{dir}/{file_name}.csv", index=False)

        elif file_type == "pth":
            torch.save(df, f"{dir}/{file_name}.pth")

    else:

        if parquet:
            df.to_parquet(f"{dir}/{file_name}.parquet", index=False)
        else:
            df.to_csv(f"{dir}/{file_name}.csv", index=False)
