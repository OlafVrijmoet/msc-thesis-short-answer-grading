
import os
import glob
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge

# classes local
from grading_models.regression.classes.Regression_Grading import Regression_Grading

# classes
from performance_tracking.classes.Measurement_Settings import Measurement_Settings
from performance_tracking.classes.Dataset import Dataset

# constants
from experiments.constants import SEEDS
from performance_tracking.constants import *

def regression():

    # loop through all cosine_similarity datasets

        # maybe as in feature_engenearing, create a data_dict?
        # save the df inside Dataset

        # dataset.split_datasets()

    # Define the base directory
    base_dir = 'data/feature_engineering/data/'

    # Use glob to get a list of all the parquet files in the base directory
    dataset_files = glob.glob(os.path.join(base_dir, '**', '*.parquet'), recursive=True)

    # Initialize an empty list to store the dictionaries
    data_list = []

    # Loop through all the dataset files
    for dataset_file in dataset_files:
        # Extract the parts from the file path
        parts = dataset_file.split(os.sep)
        
        # Extract the info from the path
        word_embed_model = parts[-6]
        abriviation_method = parts[-4]
        feature_engenearing_method = parts[-3]
        dataset_name = parts[-1].replace('.parquet', '')

        # Read the parquet file as a pandas DataFrame
        df = pd.read_parquet(dataset_file)

        # Create a dictionary with the info and DataFrame
        data_dict = {
            'word_embed_model': word_embed_model,
            'abriviation_method': abriviation_method,
            'feature_engenearing_method': feature_engenearing_method,
            'dataset_name': dataset_name,
            'df': df
        }

        # Append the dictionary to the data_list
        data_list.append(data_dict)

    # regression grading models
    regression_grading_models = [
        {
            'model_name': "Isotonic Regression",
            'model': IsotonicRegression
        },
        {
            'model_name': "Linear Regression",
            'model': LinearRegression
        },
        {
            'model_name': "Ridge Regression",
            'model': Ridge
        }
    ]

    # loop through data_list to get results
    for seed in SEEDS:

        for data in data_list:

            for regression_grading_model in regression_grading_models:

                print(f"\nregression: {data['dataset_name']}\n")

                word_embed_model = data["word_embed_model"]
                abriviation_method = data["abriviation_method"]
                feature_engenearing_method = data["feature_engenearing_method"]
                dataset_name = data["dataset_name"]

                
                if dataset_name is not "concatenated_domains":
                    dataset = Dataset(
                        dir=f"data/feature_engineering/data/{word_embed_model}/data/{abriviation_method}/{feature_engenearing_method}/data/",
                        file_name=data["dataset_name"],
                        seed=seed,
                    )

                    dataset.split_datasets()

                    model = Regression_Grading(
                        
                        # id what is being tracked
                        model=regression_grading_model["model"],
                        dataset=dataset,
                        measurement_settings=Measurement_Settings(
                            dataset_name=dataset_name,
                            embedding_seperated=True,
                            embedding_model_name=word_embed_model,
                            sentence_embedding_method=abriviation_method,
                            feature_engenearing_method=feature_engenearing_method,
                            grading_model=regression_grading_model["model_name"],
                            
                            seed_data_split=seed,

                            # inform user settings
                            print_regression=False,
                            print_classification=False,
                            
                            # save settings
                            settings_performance_tracking=ADD,
                            save_performance=True
                        ),

                        x_column="cosine_similarity",
                        y_column="normalized_points",

                        y_normalized=True,
                        
                    )
                    model.train()
                    model.test()
                    model.validation()

                if dataset_name == "concatenated_domains" or dataset_name == "concatenated_datasets":
                    
                    datasets = DATASETS
                    if dataset_name == "concatenated_domains":
                        datasets = DOMAINS

                    for dataset_name_to_split in datasets:
                        dataset = Dataset(
                            dir=f"data/feature_engineering/data/{word_embed_model}/data/{abriviation_method}/{feature_engenearing_method}/data/",
                            file_name=data["dataset_name"],
                            seed=seed,
                            left_out_dataset=dataset_name_to_split
                        )

                        dataset.split_datasets()

                        model = Regression_Grading(
                            
                            # id what is being tracked
                            model=regression_grading_model["model"],
                            dataset=dataset,
                            measurement_settings=Measurement_Settings(
                                dataset_name=dataset_name,
                                embedding_seperated=True,
                                embedding_model_name=word_embed_model,
                                sentence_embedding_method=abriviation_method,
                                feature_engenearing_method=feature_engenearing_method,
                                grading_model=regression_grading_model["model_name"],
                                left_out_dataset=dataset_name_to_split,
                                
                                seed_data_split=seed,

                                # inform user settings
                                print_regression=False,
                                print_classification=False,
                                
                                # save settings
                                settings_performance_tracking=ADD,
                                save_performance=True
                            ),

                            x_column="cosine_similarity",
                            y_column="normalized_points",

                            y_normalized=True,
                            
                        )

                        model.train()
                        model.test()
                        model.validation()
