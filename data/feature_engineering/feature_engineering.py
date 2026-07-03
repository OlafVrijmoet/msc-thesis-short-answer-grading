
import glob
import os
import pandas as pd

# services local
from data.feature_engineering.cosine_similarity.cosine_similarity import cosine_similarity

def feature_engineering():

    # Define the base directory
    base_dir = 'data/embed_sentences/data/'

    # Use glob to get all dataset file paths
    dataset_files = glob.glob(os.path.join(base_dir, '*', '*', '*', '*', '*.parquet'))
    
    # Initialize an empty list to store the dictionaries
    data_list = []

    # Loop through all the dataset files
    for dataset_file in dataset_files:
        # Extract the parts of the file path
        path_parts = dataset_file.split(os.sep)

        # Get the embed_model, abriviation_method and dataset_name from the path parts
        embed_model = path_parts[-5]
        abriviation_method = path_parts[-3]
        dataset_name = os.path.basename(dataset_file).replace('.parquet', '')

        # Read the parquet file as a pandas DataFrame
        df = pd.read_parquet(dataset_file)

        # Create a dictionary with the embed_model, abriviation_method, dataset name and DataFrame
        data_dict = {
            'embed_model': embed_model,
            'abriviation_method': abriviation_method,
            'dataset_name': dataset_name,
            'df': df
        }

        # run creation of feature engenearing
        cosine_similarity(data_dict)

        # Append the dictionary to the data_list
        data_list.append(data_dict)
