
# libaries
import os
import pandas as pd

# services
from services.import_csvs_from_dir import import_csvs_from_dir
from services.save import save

class Split_Data:

    def __init__(self, datasets_base_dir, dir_datasets, dir_new_datasets, save_existing=False):

        self.concatenated_datasets = {}
        self.existing_datasets = {}
        self.newly_split_datasets = {}

        self.save_existing = save_existing

        self.datasets_base_dir = datasets_base_dir # where the concatinated datasets are stored
        self.dir_datasets = dir_datasets # where the datasets to use are
        self.dir_new_datasets = dir_new_datasets # where the new splits should be stored
        
    def create_data_splits(self):
        
        # import dfs
        self.import_datasets()

        # if indicate save existing datasets at location of base dir
        if self.save_existing == True:
            self.save_all_in_dict(dict=self.existing_datasets, dir=self.datasets_base_dir)

        # concate all datasets
        self.concatenated_datasets["concatenated_datasets"] = pd.concat(self.existing_datasets.values())
        self.concatenated_datasets["concatenated_datasets"] = self.concatenated_datasets["concatenated_datasets"].reset_index(drop=True)

        # filter out datasets without domain
        self.concatenated_datasets["concatenated_domains"] = self.concatenated_datasets["concatenated_datasets"].dropna(subset=["domain"])
        self.concatenated_datasets["concatenated_domains"] = self.concatenated_datasets["concatenated_domains"].reset_index(drop=True)

        # save concatenated
        self.save_all_in_dict(dict=self.concatenated_datasets, dir=self.datasets_base_dir)

        # split on domains
        self.domain_split()

        # save domain split datasets
        self.save_all_in_dict(dict=self.newly_split_datasets, dir=self.dir_new_datasets)

    def import_datasets(self):

        import_csvs_from_dir(
            dict_datasets=self.existing_datasets,
            dir_datasets=self.dir_datasets
        )

    def domain_split(self):

        for name, group in self.concatenated_datasets["concatenated_domains"].groupby('domain'):
            self.newly_split_datasets[name] = group

    def save_all_in_dict(self, dict, dir):

        for name, df in dict.items():

            save(
                dir=dir,
                file_name=name,
                df=df
            )

