
import os

# classes local
from run_models.embed_words.classes.Embed_Words import Embed_Words

# classes
from classes.Dataset_Settings import Dataset_Settings
from word_embedding.models.classes.EmbeddingModel import EmbeddingModel

# services
from run_models.gensim.services.gensim_services import gensim_download, gensim_save, gensim_load

def embed_words():

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

        # "word2vec": EmbeddingModel(
        #     model_name="word2vec",
        #     download_link="word2vec-google-news-300",
        #     download_func=gensim_download,
        #     save_func=gensim_save,
        #     load_func=gensim_load
        # ),

        "conceptnet": EmbeddingModel(
            model_name="conceptnet",
            download_link="conceptnet-numberbatch-17-06-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load,
            dir_in_model_embedding="/c/en/"
        )

    }

    # loop through embedding models
    for key, model in models.items():

        model.load_model()

        # loop through data sets
        for root, dirs, files in os.walk(f"data_saved/gensim"):

            for df_name in files:

                if df_name.endswith(".csv"):

                    file_path = os.path.join(root, df_name)
                    df_name, _ = os.path.splitext(df_name)

                    print(f"move datasets into DatasetClass: {df_name}")

                    model_name = f"emebed_words_{model.model_name}"

                    dataset = Embed_Words(
                        df_name=df_name,
                        model_name=model_name, # used for dir name inside data_saved

                        language="english",

                        datasets = {
                            "gensim": Dataset_Settings(
                                df=None,
                                may_run_now=False,
                                required=True,
                                force_run=False
                            ),
                            model_name: Dataset_Settings(
                                df=None,
                                may_run_now=True,
                                required=True,
                                parquet=True,
                                name_required_dataset="gensim",
                                force_run=True
                            ),
                        },

                        embedding_model=model

                    )

                    # get dataset, process dataset, save dataset
                    dataset.run_all()
