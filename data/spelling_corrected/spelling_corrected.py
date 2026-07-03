
import os

# classes
from classes.Dataset_Settings import Dataset_Settings

# classes local
from data.spelling_corrected.classes.Correct_Spelling import Correct_Spelling

def spelling_corrected():

    # loop through all files inside splits
    for file_name in os.listdir("data/splits/data"):

        # get file name without file type for get_df
        file_name, _ = os.path.splitext(file_name)

        dataset = Correct_Spelling(
            df_name=file_name,
            model_name="spelling_corrected", # used for dir name inside data_saved

            language="english",

            datasets = {
                "standardized_splits": Dataset_Settings(
                    df=None,
                    df_name="splits",
                    base_dir="data",

                    may_run_now=False,
                    required=True,
                ),
                "spelling_corrected": Dataset_Settings(
                    df=None,
                    df_name="spelling_corrected",
                    base_dir="data",

                    may_run_now=True,
                    required=True

                ),
            },
        )

        dataset.run_all()
