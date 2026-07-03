
import os

# classes
from run_models.cosine_similarity.classes.Dataset_Cosine import Dataset_Cosine
from classes.Dataset_Settings import Dataset_Settings

def cosine_similarity(dataset_dict):

    embed_model = dataset_dict["embed_model"]
    abriviation_method = dataset_dict["abriviation_method"]

    dataset = Dataset_Cosine(
        df_name=dataset_dict["dataset_name"],
        model_name=f"cosine_similarity", # used for dir name inside data_saved

        language="english",

        datasets = {
            f"emebeded_sentences": Dataset_Settings(
                df=None,
                df_name=abriviation_method,
                base_dir=f"data/embed_sentences/data/{embed_model}/data",

                may_run_now=False,
                required=True,
                parquet=True,
                force_run=False
            ),
            f"cosine_similarity": Dataset_Settings(
                df=None,
                df_name="cosine_similarity",
                base_dir=f"data/feature_engineering/data/{embed_model}/data/{abriviation_method}",

                may_run_now=True,
                required=True,
                parquet=True,
                name_required_dataset="gensim",
                force_run=True
            ),
        },

        columns_to_add = {f"cosine_similarity": {"cosine_similarity": []}},

    )

    # get dataset, process dataset, save dataset
    dataset.get_dataset()
    dataset.process_dataset()
    dataset.add_columns() # should this not be before process_dataset?!
    dataset.save()
