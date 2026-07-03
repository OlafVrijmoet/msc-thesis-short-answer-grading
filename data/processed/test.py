
#%%
import os
import pandas as pd

# specify the path to the folder containing CSV files
folder_path = "./data/raw_data"

# create an empty list to hold dataframes
dfs = []

# iterate over all files in the folder
for file_name in os.listdir(folder_path):
    # check if the file is a CSV file
    if file_name.endswith(".csv"):
        # read the CSV file into a dataframe
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        # append the dataframe to the list
        dfs.append(df)

# concatenate all dataframes into a single dataframe
merged_df = pd.concat(dfs)

# print the merged dataframe
merged_df.dropna()


# %%
unique_domains = merged_df['domain'].unique()
print(unique_domains)

value_count = merged_df['domain'].value_counts()
value_count

# %%
import pandas as pd

# create a sample dataframe
df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David'], 
                   'Score': [90, 85, 75, 92, 80]})

# create a dictionary of dataframes for each unique value in the 'Name' column
domain_dfs = {}
for name, group in merged_df.groupby('domain'):
    domain_dfs[name] = group

# print the dataframes for each unique value in the 'Name' column
for name, df_name in domain_dfs.items():
    print(f"Dataframe for {name}:")

# %%
from spellchecker import SpellChecker

spell = SpellChecker()
text = ["Proces", "how", "are", "you", "doing"]
text_checked = [spell.correction(token) if spell.correction(token) is not None else token for token in text]
print(text)
print(text_checked)
# %%
