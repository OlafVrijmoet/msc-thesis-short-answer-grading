
# word embedders
from data.embed_words.gensim_embedding.gensim_embedding import gensim_embedding
from data.embed_words.BERT_embedding.BERT_embedding import BERT_embedding

# services
from services.get_dfs import get_dfs

# contants
from data.splits.constants import SPLITS

def embed_words():

    gensim_embedding()

    BERT_embedding()
