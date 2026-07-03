
# libaries
import os
import pandas as pd
from tqdm import tqdm

import string

# libaries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk.stem.snowball import SnowballStemmer
from spellchecker import SpellChecker

import torch

# services
from services.import_csvs_from_dir import import_csvs_from_dir
from services.save import save
from services.get_df import get_df

# classes
from classes.Process_Stages import Process_Stages

# constants
from data.splits.constants import *
from run_models.gensim.constants import GENSIM_DATA
from constants_dir.path_constants import BASIC_PROCCESSED, DATA_STAGES

# dataset porcessing
class Dataset:

    def __init__(self, df_name, model_name, datasets, language, columns_to_add={}, save_new_colums_as_torch=False) -> None:

        self.df_name = df_name
        self.model_name = model_name

        self.datasets = datasets

        self.latest_already_processed_phase = "standardized_splits" # this indicates the df that should be used for itteration 

        self.language = language

        self.columns_to_add = columns_to_add # structure {dataset_name: {column_name: [values]}}
        self.save_new_colums_as_torch = save_new_colums_as_torch

        self.spell = SpellChecker()
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    # get dataset, process dataset, save dataset
    def run_all(self):
        
        self.get_dataset()
        self.process_dataset()
        self.save()

    def get_dataset(self):

        for key, dataset in self.datasets.items():
            
            # fetch standardized_splits from special dir
            if key == "standardized_splits":

                standardized_splits_save_location = dataset["save_location"]
                self.fetch_dataset_and_replace_null(key=key, dir=f"{standardized_splits_save_location}/{self.df_name}.csv")

            else:
                                
                # check if basic processing already done, located at data_saved/basic_processed/df_name
                df_found, df_name, df = get_df(
                    dir=dataset["save_location"], 
                    file_name=self.df_name
                )
                                
                # update value of dataset done
                if dataset["force_run"] == True:
                    dataset["done"] = False
                    df_found = False
                else:
                    dataset["done"] = df_found

                print(f"df found: {df_found}, name: {key}")

                if df_found:

                    dataset["df"] = df
                    self.replace_non_with_string(key)

                    self.latest_already_processed_phase = key

                else:
                    
                    if dataset["may_run_now"] == False and dataset["required"] == True:

                        # this meas it you need a different dataset class child to run this & it has not been run yet
                        # so break run!
                        raise ValueError(f"{key} is a required datasets but is not present and it is not able to be run with this dataset class!")
                    
        # use the latest dataset and copy it to all none valued in self.datasets[key]["df"]
        for key, dataset in self.datasets.items():

            if self.datasets[key]["df"] is None or dataset["force_run"] == True:

                self.datasets[key]["df"] = self.datasets[self.latest_already_processed_phase]["df"].copy()

    def fetch_dataset_and_replace_null(self, key, dir):

        # check if dataset is parquet
        if self.datasets[key]["parquet"] == True:
            # fetch parquet
            self.datasets[key]["df"] = pd.read_parquet(dir)
        else:
            # fetch base dataset from data/splits/self.df_name
            self.datasets[key]["df"] = pd.read_csv(dir)

        self.replace_non_with_string(key)

    def replace_non_with_string(self, key):
        
        # replace Nan answers with emty string
        self.datasets[key]["df"][["student_answer", "reference_answer", "question"]] = self.datasets[key]["df"][["student_answer", "reference_answer", "question"]].fillna('')

    def process_dataset(self):

        # ensure there is something to be updated in the df
        if self.any_datasets_should_run() == True:
            
            print(f"processing starting for {self.df_name}")

            # itterate rows of df
            for index, row in tqdm(self.datasets[self.latest_already_processed_phase]["df"].iterrows(), total=self.datasets[self.latest_already_processed_phase]["df"].shape[0]):

                # process row
                processed_row_dict = self.process_row(row)
                for key, processed_row in processed_row_dict.items():
                    if key != "row":
                        self.datasets[key]["df"].loc[index] = processed_row
            
    def any_datasets_should_run(self):
        
        run_loop = False

        for key, dataset in self.datasets.items():

            if self.datasets[key]["may_run_now"] == True and self.datasets[key]["done"] == False:

                run_loop = True

                print(f"Running {key} for {self.df_name}")
            
            else:
                
                print(f"No need to run {key} for {self.df_name}")

        return run_loop

    def process_row(self, row):

        row_dict = {
            "row": row
        }

        return row_dict

    def keep_only_text(self, text: str) -> str:

        # define the regular expression pattern
        pattern = r'[^\w\s]'

        return text.replace(pattern, '')

    def strip_extra_whitespace(self, s: str) -> str:
        return " ".join(s.split())

    def strip_punctuation(self, text: str) -> str:

        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        return text
    
    def correctSpelling(self, text):

        if not text:
            return ""

        words = text.split()

        corrected_words = [self.spell.correction(word) if word != self.spell.correction(word) and self.spell.correction(word) is not None else word for word in words]
        corrected_sentence = " ".join(corrected_words)

        # Initialize the spell checker
        return corrected_sentence

    def add_columns(self):

        for name_dataset_to_add_column, conlumn_to_add in self.columns_to_add.items():

            for column_name, column_values in conlumn_to_add.items():

                if self.save_new_colums_as_torch == True:
                    print(f"{self.datasets[self.model_name]['save_location']}/{self.df_name}.pth")
                    
                    save(
                        dir=self.datasets[self.model_name]['save_location'],
                        file_name=self.df_name,
                        df=column_values,
                        file_type="pth"
                    )

                else:
                    # add column to dataset
                    self.datasets[name_dataset_to_add_column]["df"][column_name] = column_values

    def dataset_splits(self, seed, x_column_name, y_column_name):

        self.datasets[self.df_name].split_datasets(seed, x_column_name, y_column_name)

    # get attribute values
    def __getitem__(self, key):
        return getattr(self, key)
    
    # set attribute values
    def __setitem__(self, key, value):
        setattr(self, key, value)

    # save basic processed
    def save(self):

        for key, dataset in self.datasets.items():

            # skip the base df
            if key == "standardized_splits":
                continue

            # check if any new basic processing is done
            if self.datasets[key]["may_run_now"] == True and self.datasets[key]["done"] == False:
                print(f"saving new {key} phase for: {self.df_name}")
                save(
                    dir=self.datasets[key]["save_location"],
                    file_name=self.df_name,
                    df=dataset["df"],
                    parquet=self.datasets[key]["parquet"]
                )
            else:
                print(f"no saving needed because basic {key} already done on {self.df_name}")
