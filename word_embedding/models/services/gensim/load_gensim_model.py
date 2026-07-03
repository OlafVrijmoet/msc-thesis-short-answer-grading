
# libaries
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors

def gensim_download(download_link):

    return gensim_api.load(download_link)

def gensim_save(model, model_file):
    
    model.save(model_file)

def gensim_load(model_file):
    return KeyedVectors.load(model_file)
