
import os

# classes
from classes.Dataset_Settings import Dataset_Settings

# classes local
from data.BERT_ASAG_tokenization.classes.BERT_ASAG_Tokenization import BERT_ASAG_Tokenization

def BERT_tokenization():

    PyTorch_pre_trained_model_names = ["bert-base-uncased", "bert-base-cased", "distilbert-base-uncased", "distilbert-base-cased"]

    for PyTorch_pre_trained_model_name in PyTorch_pre_trained_model_names:
        
        # loop through all files inside splits
        for file_name in os.listdir("data/splits/data"):

            # get file name without file type for get_df
            file_name, _ = os.path.splitext(file_name)

            run_correct_spelling = False

            base_dir = f"data/BERT_ASAG_tokenization/data/{PyTorch_pre_trained_model_name}/data/spelling_not_corrected"
            dataset_name = "BERT_tokens"

            dataset = BERT_ASAG_Tokenization(
                df_name=file_name,
                model_name=dataset_name, # used for dir name inside data_saved

                PyTorch_pre_trained_model_name = PyTorch_pre_trained_model_name,

                language="english",

                datasets = {
                    "standardized_splits": Dataset_Settings(
                        df=None,
                        df_name="splits",
                        base_dir="data",

                        may_run_now=False,
                        required=True,
                    ),
                    dataset_name: Dataset_Settings(
                        df=None,
                        df_name=dataset_name,
                        base_dir=base_dir,

                        may_run_now=True,
                        required=True
                    ),
                },

                columns_to_add = {dataset_name: {"tokenized_for_BERT": []}}, # structure {dataset_name: {column_name: [values]}}
                save_new_colums_as_torch=True,

            )

            # get dataset, process dataset, save dataset
            dataset.get_dataset()
            dataset.process_dataset()
            dataset.add_columns()
            dataset.save()
            
            run_correct_spelling = True
            if run_correct_spelling == True:
                base_dir = f"data/BERT_ASAG_tokenization/data/{PyTorch_pre_trained_model_name}/data/spelling_corrected"
            
            dataset = BERT_ASAG_Tokenization(
                df_name=file_name,
                model_name=dataset_name, # used for dir name inside data_saved

                PyTorch_pre_trained_model_name = PyTorch_pre_trained_model_name,

                language="english",

                datasets = {
                    "spelling_corrected": Dataset_Settings(
                        df=None,
                        df_name="spelling_corrected",
                        base_dir="data",

                        may_run_now=False,
                        required=True,
                    ),
                    dataset_name: Dataset_Settings(
                        df=None,
                        df_name=dataset_name,
                        base_dir=base_dir,

                        may_run_now=True,
                        required=True,

                        force_run=True

                    ),
                },

                columns_to_add = {dataset_name: {"tokenized_for_BERT": []}}, # structure {dataset_name: {column_name: [values]}}
                save_new_colums_as_torch=True,

            )

            # get dataset, process dataset, save dataset
            dataset.get_dataset()
            dataset.process_dataset()
            dataset.add_columns()
            dataset.save()
