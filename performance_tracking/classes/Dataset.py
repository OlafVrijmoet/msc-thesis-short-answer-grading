
from sklearn.model_selection import train_test_split

# services
from services.get_df import get_df

class Dataset:

    def __init__(self,

        # to get dataset
        dir,
        file_name,

        # seed for splitting data
        seed,

        left_out_dataset=None,

        # sampling
        sample_size=None,
        sampling_group=None

    ) -> None:
        
        # split seed
        self.seed = seed
        
        # original dataset
        self.name = file_name
        self.dataset = self.get_data(dir, file_name, sample_size, sampling_group)

        # places to save the splits
        self.train = None
        self.test = None
        self.validation = None

        self.left_out_dataset = left_out_dataset

        # sampling
        self.sample_size = sample_size
        self.sampling_group = sampling_group

    # get data using dataset_dir
    def get_data(self, dir, file_name, sample_size=None, sampling_group=None):

        found, df_name, dataset = get_df(dir=dir, file_name=file_name)

        return dataset
    
    def split_datasets(self):

        if self.left_out_dataset is not None:
            
            column_to_select_from = "dataset_name"

            if self.name == "concatenated_domains":

                column_to_select_from = "domain"

            self.validation = self.dataset.query(f'{column_to_select_from} == "{self.left_out_dataset}"')
            
            print(f"\n\n***self.validation size: {len(self.validation)}, column_to_select_from: {column_to_select_from}***\n\n")

            # the remaining dataset
            remaining_data = self.dataset.query(f'{column_to_select_from} != "{self.left_out_dataset}"')

            # Split the remaining data into train and test sets
            self.train, self.test = train_test_split(remaining_data, test_size=0.2, random_state=self.seed)

        else:

            # Split the DataFrame into train (70%) and the remaining data (30%)
            self.train, temp_df = train_test_split(self.dataset, test_size=0.3, random_state=self.seed)

            # Split the remaining data further into test (20% of the original dataset) and validation (10% of the original dataset) sets
            self.test, self.validation = train_test_split(temp_df, test_size=1/3, random_state=self.seed)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
