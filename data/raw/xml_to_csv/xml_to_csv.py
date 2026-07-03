
import os

# libaries
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# classes
from data.raw.xml_to_csv.classes.Column_Data import Column_Data
from data.raw.xml_to_csv.classes.Df_info import Df_Info

# services
from data.raw.xml_to_csv.services.read_xml import read_xml

# takes a list of xml file paths, imports them and saves them at path as csv
def xml_to_csv(datasets, path_save):

    # loop through datasets
    for dataset in datasets:

        # save data
        df_info = Df_Info()
        column_data = Column_Data()

        # loop through the paths related to each dataset
        for path in dataset.paths:

            # parse the XML file
            for filename in os.listdir(path):

                if filename.endswith(".xml"):
                    tree = ET.parse(os.path.join(path, filename))
                    root = tree.getroot()

                    # read xml data
                    read_xml(root, column_data, df_info, None)

        # create df
        data = pd.DataFrame(vars(column_data))

        # save as csv
        data.to_csv(f"./{path_save}/{dataset.name}.csv", index=False)
