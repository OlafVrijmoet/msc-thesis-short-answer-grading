
import os
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge

# classes
from run_models.regressions.classses.Regression_Model import Regression_Model
from performance_tracking.classes.Dataset import Dataset

from performance_tracking.classes.Measurement_Settings import Measurement_Settings

# constants
from performance_tracking.constants import *

def experiments():

    # splitting data
    # seeds = [5, 10, 42, 60, 85]
    seeds = [42]

    embedded_datasets = [
        {   
            "embedding_model": "conceptnet", 
            "dir": "data_saved/cosine_similarity_conceptnet"},
        {   
            "embedding_model": "fasttext",
            "dir": "data_saved/cosine_similarity_fasttext"},
        {   
            "embedding_model": "glove", 
            "dir": "data_saved/cosine_similarity_glove"}
    ]

    classification_models = [
        {
            'model_name': "IsotonicRegression",
            'model': IsotonicRegression
        },
        {
            'model_name': "LinearRegression",
            'model': LinearRegression
        },
        {
            'model_name': "Ridge",
            'model': Ridge
        }
    ]

    # loop through datasets to experiement on
    for embedded_dataset in embedded_datasets:

        # loop through datasets
        for file_name in os.listdir(embedded_dataset["dir"]):
            
            name, extension = os.path.splitext(file_name)

            # loop through all seeds
            for seed in seeds:

                for classification_model in classification_models:

                    dataset = Dataset(
                        dir=embedded_dataset["dir"],
                        file_name=name,
                        seed=seed, 
                    )

                    dataset.split_datasets()

                    model = Regression_Model(
        
                        # id what is being tracked
                        embedding_seperated=True,
                        embedding_model_name=embedded_dataset["embedding_model"],
                        classfication_model_name=classification_model["model_name"],
                        dataset_name=name,
                        seed_data_split=seed,

                        # duplicates handeling
                        settings_performance_tacking=REPLACE,
                        measurement_settings=Measurement_Settings(
                            print_regression=False, 
                            print_classification=False, 
                            save_performance=True
                        ),
                        
                        dataset=dataset,
                        classification_model=classification_model["model"],
                        x_column="cosine_similarity",
                        y_column="normalized_points"
                        
                    )

                    model.train()
                    model.test()
                    model.validation()
