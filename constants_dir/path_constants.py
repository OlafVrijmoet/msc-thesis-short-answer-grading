
# paths data_stages
DATA_STAGES = "data_saved"
GENSIM_STAGE = "gensim"
BASIC_PROCCESSED = f"{DATA_STAGES}/basic_processed"
GENSIM_PROCESSED = f"{DATA_STAGES}/{GENSIM_STAGE}"

DATA_SPLIT = "data/split/data"

# paths
PROCESSED = "data/processed"
EMBED_WORDS = "word_embedding"

LEMMITIZED = "data/lemmitized_data"
RAW = "data/raw_data"
STEMMED = "data/stemmed_data"

LEMMITIZED_DATASETS = f"{LEMMITIZED}/datasets"
LEMMITIZED_DOMAIN = f"{LEMMITIZED}/domain"

RAW_DATASETS = f"{RAW}/datasets"
RAW_DOMAIN = f"{RAW}/domain"

STEMMED_DATASETS = f"{STEMMED}/datasets"
STEMMED_DOMAIN = f"{STEMMED}/domain"

# datasets to skip

# bert-based-cased:
# distilbert-base-cased: "beetle", "biology", "english", "texas", "ASAP_sas", "neural_course"
# "concatenated_domains", "concatenated_datasets"

# BERT
DATASETS_TO_SKIP = ["beetle", "english_language_arts", "concatenated_datasets"]
LEFT_OUT_DATASET_SKIP = ["biology", "english_language_arts", "english", "neural_networks", "science", "beetle", "neural_course", "sciEntsBank", "texas"]

# GPT:
# DATASETS_TO_SKIP = ["beetle", "english", "texas", "biology", "ASAP_sas", "neural_course", "english_language_arts", "science", "neural_networks"]
# LEFT_OUT_DATASET_SKIP = ["ASAP_sas", "beetle", "neural_course", "sciEntsBank", "texas", "english", "english_language_arts", "neural_networks", "science"]
