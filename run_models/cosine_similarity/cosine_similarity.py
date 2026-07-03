
import os

# classes
from run_models.cosine_similarity.classes.Dataset_Cosine import Dataset_Cosine
from classes.Dataset_Settings import Dataset_Settings

def cosine_similarity():

    # nog een toevoegen: "embed_words_word2vec"

    # list all models
    all_models = ["fasttext", "glove", "conceptnet"]

    for model_name in all_models:

        # loop through data sets
        for root, dirs, files in os.walk(f"data_saved/gensim"):

            for df_name in files:

                if df_name.endswith(".csv"):

                    file_path = os.path.join(root, df_name)
                    df_name, _ = os.path.splitext(df_name)

                    print(f"move datasets into DatasetClass: {df_name}")

                    dataset = Dataset_Cosine(
                        df_name=df_name,
                        model_name=f"cosine_similarity_{model_name}", # used for dir name inside data_saved

                        language="english",

                        datasets = {
                            f"emebed_words_{model_name}": Dataset_Settings(
                                df=None,
                                may_run_now=False,
                                required=True,
                                parquet=True,
                                force_run=False
                            ),
                            f"cosine_similarity_{model_name}": Dataset_Settings(
                                df=None,
                                may_run_now=True,
                                required=True,
                                parquet=True,
                                name_required_dataset="gensim",
                                force_run=True
                            ),
                        },

                        columns_to_add = {f"cosine_similarity_{model_name}": {"cosine_similarity": []}},

                    )

                    # get dataset, process dataset, save dataset
                    dataset.get_dataset()
                    dataset.process_dataset()
                    dataset.add_columns()
                    dataset.save()
