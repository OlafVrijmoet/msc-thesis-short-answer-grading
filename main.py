
# services
from services.run_phase import run_phase

# constants
from constants import *

def main():

    # raw data
    run_phase(RAW_PHASE)

    # standardize data
    run_phase(STANDARDIZE_PHASE)

    # # process text - !!!OUTDATED!!!
    # run_phase(PROCESS_TEXT_PHASE)

    # create datasets splits, example: based on domains
    run_phase(SPLIT_DATA)

    run_phase(SPELLING_CORRECTED)

    run_phase(EMBED_WORDS)

    run_phase(EMBED_SENTENCES)

    run_phase(BERT_TOKENIZATION_ASAG)

    run_phase(FEATURE_ENGENERING)

    run_phase(REGRESSION)

    run_phase(API)

    run_phase(BERT)

    # updated:

    # # embed words
    # run_phase(EMBED_WORDS)

    # # run Gensim models
    # run_phase(GENSIM)

    # # run Embedding models
    # run_phase(EMBEDDING)

    # # add cosine similarity
    # run_phase(COSINE_SIMILARITY)

    # # run experiments where embedding and classification are seperate
    # run_phase(CLASSIFICATION_EXPERIMENTS)

if __name__ == "__main__":

    main()
