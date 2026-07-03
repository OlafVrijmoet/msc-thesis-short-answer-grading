
# libaries
import pandas as pd

# services
from data.raw.xml_to_csv.xml_to_csv import xml_to_csv

# classes
from data.raw.xml_to_csv.classes.Xml_Data_Info import Xml_Data_Info

# constants
from data.raw.constants import *

def all_to_csv():

    # defining datasets
    beetle = Xml_Data_Info("beetle", "beetle", PATHS_BEETLE)
    sciEntsBank = Xml_Data_Info("sciEntsBank", "sciEntsBank", PATHS_SCI_ENTS_BANK)
    datasets = [beetle, sciEntsBank]

    # beetle & sciEntsBank
    xml_to_csv(datasets, SAVE_RAW)

    # tsv to csv
    # ASAP
    ASAP_sas = pd.read_csv('data/raw/data/ASAP_sas/ASAP_sas.tsv', sep='\t')
    ASAP_sas.to_csv(f"./{SAVE_RAW}/ASAP_sas.csv", index=False)
