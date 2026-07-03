
import os

# from word_embedding.models.constants import *

from data.embed_words.gensim_embedding.constants import GENSIM_MODEL_DIR

class Gensim_Embedding_Model:

    def __init__(self, model_name, download_link, download_func, save_func, load_func, dir_in_model_embedding = ""):
        self.model_name = model_name
        self.download_link = download_link
        self.download_func = download_func
        self.save_func = save_func
        self.load_func = load_func
        self.model = None
        
        self.dir_in_model_embedding = dir_in_model_embedding

    def load_model(self):
        # Create the directory if it doesn't exist
        os.makedirs(f"{GENSIM_MODEL_DIR}/{self.model_name}", exist_ok=True)

        model_file = os.path.join(f"{GENSIM_MODEL_DIR}/{self.model_name}", f"{self.model_name}.bin")

        # Download and save the model if it's not already downloaded
        if not os.path.isfile(model_file):
            print(f"Downloading {self.model_name}...")
            self.model = self.download_func(self.download_link)
            self.save_func(self.model, model_file)
            print(f"{self.model_name} downloaded and saved.")
        else:
            print(f"{self.model_name} already exists. Loading from file.")

        # Load the model from the local file
        self.model = self.load_func(model_file)

    # get attribute values
    def __getitem__(self, key):
        return getattr(self, key)
