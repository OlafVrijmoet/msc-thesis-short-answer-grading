
import torch
from torch.utils.data import Dataset, DataLoader

# classes
from performance_tracking.classes.Dataset import Dataset as Dataset_local

# services
from services.get_df import get_df

class Dataset_Torch(Dataset_local):

    def __init__(self, dir, file_name, seed, batch_size, left_out_dataset=None, sample_size=None, sampling_group=None) -> None:
        super().__init__(dir, file_name, seed, left_out_dataset, sample_size, sampling_group)

        self.batch_size = batch_size

        self.train_dataloader = None
        self.test_dataloader = None
        self.validation_dataloader = None

    def get_data(self, dir, file_name, sample_size=None, sampling_group=None):

        # get df
        found, df_name, dataset = get_df(dir=dir, file_name=file_name, know_type="csv")
        found_pth, df_name_pth, dataset_pth = get_df(dir=dir, file_name=file_name, know_type="pth")
        
        if dataset is None or dataset_pth is None:

            return None
    
        dataset["tokenized_for_BERT"] = dataset_pth

        if sample_size is not None and len(dataset) > sample_size:

            print("\n\n *** Sampling ***")
            print(f"length full dataset: {len(dataset)}")
            
            # if each groups has to be sampled equally
            if sampling_group is not None:
                # Calculate the number of samples per group
                unique_groups = dataset[sampling_group].nunique()
                samples_per_group = sample_size // unique_groups

                # Sample the specified number of samples from each group
                dataset = dataset.groupby(sampling_group).apply(lambda x: x.sample(min(len(x), samples_per_group))).reset_index(drop=True)

            # if there has to be random sampling
            else:

                # Sample a random subset of rows from the dataset without replacement
                dataset = dataset.sample(n=sample_size, replace=False, random_state=self.seed)
            
            print(f"length sampled dataset: {len(dataset)}")

        return dataset

    def init_dataloaders(self):

        # defining training, test and validation sets
        train_dataset = ASAGDataset(self.train)
        test_dataset = ASAGDataset(self.test)
        validation_dataset = ASAGDataset(self.validation)

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=9, pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=9, pin_memory=True)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=9, pin_memory=True)

class ASAGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        normalized_points = float(row["assigned_points"]) / float(row["max_points"])

        encoded = row["tokenized_for_BERT"]
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        # token_type_ids = encoded.get("token_type_ids", None)
        # if token_type_ids is not None:
        #     token_type_ids = token_type_ids.squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
            "normalized_points": torch.tensor(normalized_points, dtype=torch.float32),
            "assigned_points": torch.tensor(row["assigned_points"], dtype=torch.float32),
            "max_points": torch.tensor(row["max_points"], dtype=torch.float32),
        }
