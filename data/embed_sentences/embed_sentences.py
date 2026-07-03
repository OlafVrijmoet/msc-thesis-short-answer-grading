
import os
import glob
import pandas as pd
import numpy as np

# services
from services.save import save
from services.string_array import array_to_str, str_to_array

# constants local
from data.embed_sentences.constants import ABBRIVIATION_METHODS

def embed_sentences():

    # get datasets
        # loop through all folders in data/embed_words/data/
        # data/embed_words/data/{model_name}"
            # get all dfs for each embedding model: model_name
    
    # Define the directory where the data is located
    base_dir = 'data/embed_words/data/'

    # Use glob to get a list of all the subdirectories
    embed_model_dirs = glob.glob(os.path.join(base_dir, '*'))

    # Initialize an empty list to store the dictionaries
    data_list = []

    # Loop through all the embed_model directories
    for embed_model_dir in embed_model_dirs:
        # Extract the embed_model from the directory path
        embed_model = os.path.basename(embed_model_dir)

        # Use glob to get a list of all the .parquet files in the embed_model directory
        dataset_files = glob.glob(os.path.join(embed_model_dir, 'data', '*.parquet'))

        # Loop through all the dataset files
        for dataset_file in dataset_files:
            # Extract the dataset name from the file path (without file extension)
            dataset_name = os.path.basename(dataset_file).replace('.parquet', '')

            # Read the parquet file as a pandas DataFrame
            df = pd.read_parquet(dataset_file)

            # Create a dictionary with the embed_model, dataset name and DataFrame
            data_dict = {
                'embed_model': embed_model,
                'dataset_name': dataset_name,
                'df': df
            }

            # Append the dictionary to the data_list
            data_list.append(data_dict)

    # Loop through all the datasets
    for data in data_list:
        
        # Create a copy of the original DataFrame
        df = data['df'].copy()
        
        # Add or average the embedded words
        for abriviation_method_name, abriviation_method in ABBRIVIATION_METHODS.items():

            # Create a copy of the DataFrame for each operation
            df_temp = df.copy()
            
            if abriviation_method == ABBRIVIATION_METHODS["avg"]:
                # Assuming the embeddings are in 'student_answer' and 'reference_answer' columns
                # Apply mean to the 'student_answer' column
                df_temp['student_answer'] = df['student_answer'].apply(
                    lambda x: np.mean(str_to_array(x), axis=0) if np.array(str_to_array(x)).size != 0 else str_to_array(x)
                )
                
                # Apply mean to the 'reference_answer' column
                df_temp['reference_answer'] = df['reference_answer'].apply(
                    lambda x: np.mean(str_to_array(x), axis=0) if np.array(str_to_array(x)).size != 0 else str_to_array(x)
                )
            
            elif abriviation_method == ABBRIVIATION_METHODS["add"]:
                # Apply sum to the 'student_answer' column
                df_temp['student_answer'] = df['student_answer'].apply(
                    lambda x: np.sum(str_to_array(x), axis=0)
                )
                
                # Apply sum to the 'reference_answer' column
                df_temp['reference_answer'] = df['reference_answer'].apply(
                    lambda x: np.sum(str_to_array(x), axis=0)
                )
            
            else:
                print(f"abriviation method {abriviation_method_name} does not exist!")
                continue
            
            # define save directory variables
            embedding_model = data["embed_model"]
            df_name = data["dataset_name"]

            # Define the save directory
            save_dir = f"data/embed_sentences/data/{embedding_model}/data/{abriviation_method_name}/data"

            # save df
            save(
                dir=save_dir, 
                file_name=df_name, 
                df=df_temp,
                parquet=True
            )
