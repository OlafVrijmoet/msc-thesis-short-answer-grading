
# save
SAVE_PROCESSED = "data/processed/data"

BASE_DIR_RAW = f"{SAVE_PROCESSED}/raw_data"
BASE_DIR_STEMMED = f"{SAVE_PROCESSED}/stemmed_data"
BASE_DIR_LEMMITIZED = f"{SAVE_PROCESSED}/lemmitized_data"

DF_RAW = f"{BASE_DIR_RAW}/datasets"
DF_STEMMED = f"{BASE_DIR_STEMMED}/datasets"
DF_LEMMITIZED = f"{BASE_DIR_LEMMITIZED}/datasets"

DOMAIN_DF_RAW = f"{BASE_DIR_RAW}/domain"
DOMAIN_DF_STEMMED = f"{BASE_DIR_STEMMED}/domain"
DOMAIN_DF_LEMMITIZED = f"{BASE_DIR_LEMMITIZED}/domain"

# the standardized csv datasets
STANDARDIZED_BASE = "data/standardized/data"

ASAP_SAS_STANDARDIZED = "data/standardized/data/ASAP_sas.csv"
BEETLE_STANDARDIZED = "data/standardized/data/beetle.csv"
NN_COURSE_STANDARDIZED = "data/standardized/data/neural_course.csv"
SCI_ENTS_BANK_STANDARDIZED = "data/standardized/data/sciEntsBank.csv"
TEXAS_STANDARDIZED = "data/standardized/data/Texas.csv"

# stages
CHAPTER_1 = "Import datasets"
CHAPTER_2 = "Load datasets into class"
CHAPTER_3 = "Text processing & saving"
