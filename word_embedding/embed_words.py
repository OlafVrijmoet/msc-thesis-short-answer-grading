
# libaries
import os
import pandas as pd
import gensim.downloader as gensim_api

# services
from word_embedding.models.services.gensim.load_gensim_model import gensim_download, gensim_save, gensim_load
from word_embedding.models.services.gensim.embed_text_gensim import embed_text_gensim

# classes
from word_embedding.classes.Embed_Words import Embed_Words
from word_embedding.models.classes.EmbeddingModel import EmbeddingModel

# print
from services.printing.print_chapter import print_sub_chapter_start, print_sub_chapter_end

# constants
from constants_dir.path_constants import PROCESSED, STEMMED
from word_embedding.constants import WORD_EMBEDDING, CHAPTER_1, CHAPTER_2

def embed_words():

    print_sub_chapter_start(CHAPTER_1)

    fasttext = EmbeddingModel(
        model_name="fasttext",
        download_link="fasttext-wiki-news-subwords-300",
        download_func=gensim_download,
        save_func=gensim_save,
        load_func=gensim_load
    )

    glove = EmbeddingModel(
        model_name="glove",
        download_link="glove-wiki-gigaword-300",
        download_func=gensim_download,
        save_func=gensim_save,
        load_func=gensim_load
    )

    # word2vec = EmbeddingModel(
    #     model_name="word2vec",
    #     download_link="word2vec-google-news-300",
    #     download_func=gensim_download,
    #     save_func=gensim_save,
    #     load_func=gensim_load
    # )

    conceptnet = EmbeddingModel(
        model_name="conceptnet",
        download_link="conceptnet-numberbatch-17-06-300",
        download_func=gensim_download,
        save_func=gensim_save,
        load_func=gensim_load
    )

    fasttext.load_model()
    glove.load_model()
    # word2vec.load_model()
    conceptnet.load_model()

    # word2vec
    models = [fasttext, glove, conceptnet]

    print("start loop")

    # loop through embedding models
    for model in models:

        # loop through stemmed datasets
        for root, dirs, files in os.walk(f"{PROCESSED}/{STEMMED}"):

            for file in files:
                if file.endswith(".csv"):

                    print_sub_chapter_start(CHAPTER_1)

                    file_path = os.path.join(root, file)
                    file_name, _ = os.path.splitext(file)

                    print(f"Processing CSV file: {file_name}")

                    embed_df = Embed_Words(

                        name_df=file_name,
                        name_model=model.model_name,

                        df = pd.read_csv(file_path),

                        model=model,

                        embed_word=embed_text_gensim,

                        save_path=f"{WORD_EMBEDDING}/{STEMMED}",

                    )
                    
                    print_sub_chapter_end(CHAPTER_1)
                    print_sub_chapter_start(CHAPTER_2)
                    embed_df.embed_df()
                    print(f"save {file_name}")
                    embed_df.save()
                    print(f"save {file_name} done")
                    print_sub_chapter_end(CHAPTER_2)
