
import os
import pandas as pd

# services
from services.import_csvs_from_dir import import_csvs_from_dir

# classes
from classes.Dataset import Dataset

# delete!!!
class Datasets:

    def __init__(self, force_re_run, base_dir, model_name, progress_stages, language) -> None:

        self.force_re_run = force_re_run # this forces the entire pre-processing to be re-run if true even though there is already a dataset processed for a stage

        self.base_dir = base_dir # where the existing datasets are stored

        self.model_name = model_name

        self.progress_stages = progress_stages # detimes what needs to be run in the loop

        self.datasets = {}

        language=language

    def get_datasets(self):

        df_names = []

        # get all file names in a dir
        for file in os.listdir(self.base_dir):
            filename, file_extension = os.path.splitext(file)
            df_names.append(filename)

        # move datasets into DatasetClass
        for df_name in df_names:
            print(df_name)
            self.datasets[df_name] = Dataset(
                df_name=df_name,
                model_name=self.model_name,

                language="english",

                process_stages=self.progress_stages
            )

            self.datasets[df_name].get_dataset()
        
            self.datasets[df_name].process_dataset()
            
            self.datasets[df_name].save()

# what do I need for all datasets
    # get them
    # keep track of what stage they are at
