
# libaries
import os
import pandas as pd

# classes
from data.processed.classes.Split_Data import Split_Data

# constants
from data.processed.constants import *

# seems to be deleted!!!
def split_data():

    raw_data_split = Split_Data(
        datasets_base_dir=BASE_DIR_RAW,
        dir_datasets=DF_RAW,
        dir_new_datasets=DOMAIN_DF_RAW
    )

    stemmed_data_split = Split_Data(
        datasets_base_dir=BASE_DIR_STEMMED,
        dir_datasets=DF_STEMMED,
        dir_new_datasets=DOMAIN_DF_STEMMED
    )

    lemmitized_data_split = Split_Data(
        datasets_base_dir=BASE_DIR_LEMMITIZED,
        dir_datasets=DF_LEMMITIZED,
        dir_new_datasets=DOMAIN_DF_LEMMITIZED
    )

    raw_data_split.create_data_splits()
    stemmed_data_split.create_data_splits()
    lemmitized_data_split.create_data_splits()
