
import os

# classes
from data.embed_words.BERT_embedding.classes.BERT_Embedding import BERT_Embedding
from classes.Dataset_Settings import Dataset_Settings

# constants
from data.splits.constants import SPLITS

def BERT_embedding():

    # get all datasets from split
    df_names = []

    # get all file names in a dir
    for file in os.listdir(SPLITS):
        filename, file_extension = os.path.splitext(file)
        df_names.append(filename)

    # loop through all datasets
    for df_name in df_names:

        BERT_embed = BERT_Embedding(
            df_name=df_name,
            model_name="BERT", # used for dir name inside data_saved

            language="english",

            datasets = {
                "standardized_splits": Dataset_Settings(
                    df=None,
                    df_name="splits",
                    base_dir="data",

                    may_run_now=False,
                    required=True,
                ),
                "BERT": Dataset_Settings(
                    df=None,
                    df_name="BERT",
                    base_dir="data/embed_words/data",

                    may_run_now=True,
                    required=True,
                    parquet=True,
                    name_required_dataset="standardized_splits",
                    force_run=False
                ),
            },

            pre_trained_BERT_model_name="bert-base-uncased"

        )

        BERT_embed.run_all()
