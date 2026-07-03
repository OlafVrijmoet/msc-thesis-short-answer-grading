
# libaries
import os
import gensim.downloader as gensim_api
import pandas as pd

# classes
from classes.Dataset_Settings import Dataset_Settings

# local classes
from data.embed_words.gensim_embedding.classes.Dataset_Gensim import Dataset_Gensim
from data.embed_words.gensim_embedding.classes.Gensim_Embedding_Model import Gensim_Embedding_Model
from data.embed_words.gensim_embedding.classes.Gensim_Embedding import Gensim_Embedding

# local services
from data.embed_words.gensim_embedding.services.Gensim_services import gensim_download, gensim_save, gensim_load

# print
from services.printing.print_chapter import print_sub_chapter_start, print_sub_chapter_end

# constants
from data.splits.constants import SPLITS
from word_embedding.models.constants import *
from run_models.gensim.constants import GENSIM_DATA

# local constants
from data.embed_words.constants import EMBED_WORDS
from data.embed_words.gensim_embedding.constants import GENSIM_MODEL_NAMES

def gensim_embedding():

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
                    df_name="splits",
                    base_dir="data",

                    may_run_now=False,
                    required=True,
                ),
                "basic_processed": Dataset_Settings(
                    df=None,
                    df_name="basic_processed",
                    base_dir="data",

                    may_run_now=True,
                    required=True
                ),
                "gensim": Dataset_Settings(
                    df=None,
                    df_name="gensim_embedding",
                    base_dir="data/embed_words",

                    may_run_now=True,
                    required=True
                ),
            },

        )

        # get dataset, process dataset, save dataset
        datasets[df_name].run_all()

    # models
    models = {
        GENSIM_MODEL_NAMES["fasttext"]: Gensim_Embedding_Model(
            model_name=GENSIM_MODEL_NAMES["fasttext"],
            download_link="fasttext-wiki-news-subwords-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        ),

        GENSIM_MODEL_NAMES["glove"]: Gensim_Embedding_Model(
            model_name=GENSIM_MODEL_NAMES["glove"],
            download_link="glove-wiki-gigaword-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        ),

        # "word2vec": Gensim_Embedding_Model(
        #     model_name="word2vec",
        #     download_link="word2vec-google-news-300",
        #     download_func=gensim_download,
        #     save_func=gensim_save,
        #     load_func=gensim_load
        # ),

        GENSIM_MODEL_NAMES["conceptnet"]: Gensim_Embedding_Model(
            model_name=GENSIM_MODEL_NAMES["conceptnet"],
            download_link="conceptnet-numberbatch-17-06-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load,
            dir_in_model_embedding="/c/en/"
        )

    }

    # loop through embedding models
    for key, model in models.items():
        
        # download / fetch model
        model.load_model()

        # loop through data sets
        for root, dirs, files in os.walk(f"data_saved/gensim"):

            for df_name in files:

                if df_name.endswith(".csv"):

                    file_path = os.path.join(root, df_name)
                    df_name, _ = os.path.splitext(df_name)

                    print(f"move datasets into DatasetClass: {df_name}")

                    model_name = f"{model.model_name}"

                    dataset = Gensim_Embedding(
                        df_name=df_name,
                        model_name=model_name, # used for dir name inside data_saved

                        language="english",

                        datasets = {
                            "gensim": Dataset_Settings(
                                df=None,
                                df_name="gensim_embedding",
                                base_dir="data/embed_words",

                                may_run_now=False,
                                required=True
                            ),
                            model_name: Dataset_Settings(
                                df=None,
                                df_name=model_name,
                                base_dir="data/embed_words/data",

                                may_run_now=True,
                                required=True,
                                parquet=True,
                                name_required_dataset="gensim",
                                force_run=False
                            ),
                        },

                        embedding_model=model

                    )

                    # get dataset, process dataset, save dataset
                    dataset.run_all()
