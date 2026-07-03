
# libaries
import nltk
import pandas as pd

# classes
from data.processed.classes.Process_Text import Process_Text

# print
from services.printing.print_chapter import print_sub_chapter_start, print_sub_chapter_end

# constants
from data.processed.constants import *

def process_text():

    # needed for lemmatization
    nltk.download('wordnet')

    print_sub_chapter_start(CHAPTER_1)

    # import data
    ASAP_sas = pd.read_csv(ASAP_SAS_STANDARDIZED)
    beetle = pd.read_csv(BEETLE_STANDARDIZED)
    neural_course = pd.read_csv(NN_COURSE_STANDARDIZED)
    sciEntsBank = pd.read_csv(SCI_ENTS_BANK_STANDARDIZED)
    texas = pd.read_csv(TEXAS_STANDARDIZED)

    print_sub_chapter_end(CHAPTER_1)

    print_sub_chapter_start(CHAPTER_2)

    # setup
    ASAP_sas_class = Process_Text(
        df = ASAP_sas,
        name = "ASAP_sas"
    )
    
    beetle_class = Process_Text(
        df = beetle,
        name = "beetle"
    )

    neural_class = Process_Text(
        df = neural_course,
        name = "neural_course"
    )

    sciEntsBank_class = Process_Text(
        df = sciEntsBank,
        name = "sciEntsBank"
    )

    texas_class = Process_Text(
        df = texas,
        name = "texas"
    )

    print_sub_chapter_end(CHAPTER_2)
    print_sub_chapter_start(CHAPTER_3)

    # run text processing & saving
    ASAP_sas_class.process_text()
    beetle_class.process_text()
    neural_class.process_text()
    sciEntsBank_class.process_text()
    texas_class.process_text()

    print_sub_chapter_end(CHAPTER_3)
