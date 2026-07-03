
# libaries
import os
import gensim.downloader as gensim_api
import pandas as pd

# classes
from word_embedding.classes.Embed_Words import Embed_Words
from word_embedding.models.services.gensim.embed_text_gensim import embed_text_gensim
from classes.Datasets import Datasets
from run_models.gensim.classes.Process_Stages_Gensim import Process_Stages_Gensim
from word_embedding.models.classes.EmbeddingModel import EmbeddingModel
from run_models.gensim.classes.Dataset_Gensim import Dataset_Gensim
from classes.Dataset_Settings import Dataset_Settings

# services
from run_models.gensim.services.gensim_services import gensim_download, gensim_save, gensim_load

# print
from services.printing.print_chapter import print_sub_chapter_start, print_sub_chapter_end

# constants
from data.splits.constants import *
from word_embedding.models.constants import *
from run_models.gensim.constants import GENSIM_DATA

def gensim():

    datasets = {}
    df_names = []

    # get all file names in a dir
    for file in os.listdir(SPLITS):
        filename, file_extension = os.path.splitext(file)
        df_names.append(filename)

    # move datasets into DatasetClass
    for df_name in df_names:
        print(f"move datasets into DatasetClass: {df_name}")

        datasets[df_name] = Dataset_Gensim(
            df_name=df_name,
            model_name="gensim", # used for dir name inside data_saved

            language="english",

            datasets = {
                "standardized_splits": Dataset_Settings(
                    df=None,
                    may_run_now=False,
                    required=True
                ),
                "basic_processed": Dataset_Settings(
                    df=None,
                    may_run_now=True,
                    required=True
                ),
                "gensim": Dataset_Settings(
                    df=None,
                    may_run_now=True,
                    required=True
                ),
            },

        )

        # get dataset, process dataset, save dataset
        datasets[df_name].run_all()

    # normalize points for each dataset
    for df_name in df_names:
        print(f"normalizing points for dataset: {df_name}")

        df = pd.read_csv(f"data_saved/gensim/{df_name}.csv")

        # round values to howl int numbers
        df["assigned_points"] = df["assigned_points"].round()
        df["assigned_points"] = df["assigned_points"].astype(int)

        # add normalized points
        df["normalized_points"] = df["assigned_points"] / df["max_points"]

        df.to_csv(f"data_saved/gensim/{df_name}.csv", index=False)

    # models
    models = {
        "fasttext": EmbeddingModel(
            model_name="fasttext",
            download_link="fasttext-wiki-news-subwords-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        ),

        "glove": EmbeddingModel(
            model_name="glove",
            download_link="glove-wiki-gigaword-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        ),

        "word2vec": EmbeddingModel(
            model_name="word2vec",
            download_link="word2vec-google-news-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        ),

        "conceptnet": EmbeddingModel(
            model_name="conceptnet",
            download_link="conceptnet-numberbatch-17-06-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        )

    }

    # loop through embedding models
    for key, model in models.items():

        model.load_model()
