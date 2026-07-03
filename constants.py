
# data processing
from data.raw.all_to_csv import all_to_csv
from data.standardized.standardize import standardize
from data.processed.process_text import process_text
from data.splits.split_data import split_data
from data.spelling_corrected.spelling_corrected import spelling_corrected

from data.embed_words.embed_words import embed_words
from data.embed_sentences.embed_sentences import embed_sentences
from data.feature_engineering.feature_engineering import feature_engineering
from data.BERT_ASAG_tokenization.BERT_tokenization import BERT_tokenization

# grading
from grading_models.regression.regression import regression
from grading_models.api.api import api
from grading_models.BERT.BERT import bert

# data processing for models
# from run_models.gensim.gensim import gensim
# from run_models.embed_words.embed_words import embed_words
# from run_models.cosine_similarity.cosine_similarity import cosine_similarity
# from experiments.experiments import experiments

# classes
from classes.Phase_Settings import Phase_Settings

# what to run
RAW_PHASE = Phase_Settings(
    name="Raw", 
    function=all_to_csv,
    run=False
)

STANDARDIZE_PHASE = Phase_Settings(
    name="Standardize", 
    function=standardize,
    run=False
)

PROCESS_TEXT_PHASE = Phase_Settings(
    name="Process text", 
    function=process_text,
    run=False
)

SPLIT_DATA = Phase_Settings(
    name="Split data", 
    function=split_data,
    run=False
)

SPELLING_CORRECTED = Phase_Settings(
    name="spelling Corrected",
    function=spelling_corrected,
    run=False
)

EMBED_WORDS = Phase_Settings(
    name="Embed words", 
    function=embed_words,
    run=False
)

EMBED_SENTENCES = Phase_Settings(
    name="Embed sentences", 
    function=embed_sentences,
    run=False
)

BERT_TOKENIZATION_ASAG = Phase_Settings(
    name="BERT tokenization for ASAG", 
    function=BERT_tokenization,
    run=False
)

FEATURE_ENGENERING = Phase_Settings(
    name="feature engenearing", 
    function=feature_engineering,
    run=False
)

REGRESSION = Phase_Settings(
    name="regression", 
    function=regression,
    run=False
)

API = Phase_Settings(
    name="api", 
    function=api,
    run=False
)

BERT = Phase_Settings(
    name="bert", 
    function=bert,
    run=True
)

# # delete
# EMBED_WORDS = Phase_Settings(
#     name="Embed words", 
#     function=embed_words,
#     run=False
# )

# GENSIM = Phase_Settings(
#     name="sensim_models", 
#     function=gensim,
#     run=False
# )

# EMBEDDING = Phase_Settings(
#     name="sensim_models", 
#     function=embed_words,
#     run=False
# )

# COSINE_SIMILARITY = Phase_Settings(
#     name="cosine_similarity", 
#     function=cosine_similarity,
#     run=False
# )

# CLASSIFICATION_EXPERIMENTS = Phase_Settings(
#     name="classification experiments", 
#     function=experiments,
#     run=True
# )

# paths
# LEMMITIZED = "data/lemmitized_data"
# RAW = "data/raw_data"
# STEMMED = "data/stemmed_data"

# LEMMITIZED_DATASETS = f"{LEMMITIZED}/datasets"
# LEMMITIZED_DOMAIN = f"{LEMMITIZED}/domain"

# RAW_DATASETS = f"{RAW}/datasets"
# RAW_DOMAIN = f"{RAW}/domain"

# STEMMED_DATASETS = f"{STEMMED}/datasets"
# STEMMED_DOMAIN = f"{STEMMED}/domain"
