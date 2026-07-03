
# libaries
import os
import pandas as pd

# classes
from data.processed.classes.Split_Data import Split_Data

# constants
from data.processed.constants import *
from data.splits.constants import *

def split_data():

    data_split = Split_Data(
        datasets_base_dir=SPLITS,
        dir_datasets=STANDARDIZED_BASE,
        dir_new_datasets=SPLITS,

        save_existing=True
    )

    data_split.create_data_splits()

    # normalize points for each dataset
    df_names = []
    # get all file names in a dir
    for file in os.listdir(SPLITS):
        filename, file_extension = os.path.splitext(file)
        df_names.append(filename)

    # normalize points for each dataset
    for df_name in df_names:
        print(f"normalizing points for dataset: {df_name}")

        df = pd.read_csv(f"{SPLITS}/{df_name}.csv")

        # round values to howl int numbers
        df["assigned_points"] = df["assigned_points"].round()
        df["assigned_points"] = df["assigned_points"].astype(int)

        # add normalized points
        df["normalized_points"] = df["assigned_points"] / df["max_points"]

        df.to_csv(f"{SPLITS}/{df_name}.csv", index=False)
