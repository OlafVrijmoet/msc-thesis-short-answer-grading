
import os
from performance_tracking.classes.Measurement_Settings import Measurement_Settings
from performance_tracking.constants import *

from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from datetime import datetime

# classes
from performance_tracking.classes.Measurement_Settings import Measurement_Settings
from performance_tracking.classes.Dataset_Torch import Dataset_Torch

# classes local
from grading_models.BERT.classes.Py_Torch import Py_Torch

# services
from services.get_df import get_df

# constants
from experiments.constants import SEEDS, SHOTS
from performance_tracking.constants import *
from constants_dir.path_constants import DATASETS_TO_SKIP, LEFT_OUT_DATASET_SKIP

def bert():

    model_name = "bert-base-cased"
    
    base_dir = f"data/BERT_ASAG_tokenization/data/{model_name}/data/spelling_corrected/BERT_tokens/data"
    
    for SEED in SEEDS:

        # Use os.listdir to get a list of all files in the directory
        for filename in os.listdir(base_dir):

            # Concatenate the directory name with the filename to get the full path
            full_path = os.path.join(f"{base_dir}/", filename)
            
            # Check if the path is a file
            if os.path.isfile(full_path):

                df_name, file_extenstion = os.path.splitext(filename)
                
                # prevent from running every dataset twice
                if file_extenstion == ".pth":
                    continue

                # if df_name in DATASETS_TO_SKIP:
                #     continue
                
                unfrozen_layers_count = None
                description = "seplling_corrected_sample_2000"
                if unfrozen_layers_count is not None:
                    description = f"{description}_unforzen_layers_{unfrozen_layers_count}"

                # if df_name != "concatenated_datasets":

                #     continue

                # if df_name != "concatenated_domains":

                if df_name == "concatenated_domains":
                    
                    # for re-runs because this one already ran
                    # if df_name in ["concatenated_datasets"]:
                    #     continue

                    print(f"\n\n*** start time: {datetime.now()} ***")
                    print(f"Running {model_name} on {df_name}")

                    # Now you can do whatever you want with the file
                    dataset = Dataset_Torch(
                        dir = base_dir,
                        file_name = df_name,
                        seed = SEED,
                        batch_size=128,
                        sample_size=2000,
                        sampling_group="dataset_name" if df_name == "concatenated_datasets" else None
                    )
                    saved_model_dir = f"grading_models/BERT/saved_models/{model_name}/{description}/{dataset['name']}"
                    
                    dataset.split_datasets()
                    print(f"length train: {len(dataset['train'])}")
                    dataset.init_dataloaders()

                    dataset_grading = Py_Torch(
                        # parent
                        model=BertForSequenceClassification.from_pretrained(model_name, num_labels=1),
                        dataset=dataset,
                        measurement_settings=Measurement_Settings(
                            dataset_name=dataset["name"],
                            embedding_seperated=False,
                            sentence_embedding_method=None,
                            feature_engenearing_method=None,

                            embedding_model_name=model_name,
                            grading_model=model_name,
                            
                            seed_data_split=42,

                            description = description,

                            # inform user settings
                            print_regression=True,
                            print_classification=True,
                            
                            # save settings
                            settings_performance_tracking=ADD,
                            save_performance=True
                        ),

                        # child
                        y_column="assigned_points",

                        y_normalized=False, # idd not normalized because measurments are doen on non normalized values!

                        lr = 2e-5,
                        saved_model_dir = saved_model_dir,
                        epochs_to_run = 5,

                        unfrozen_layers_count=unfrozen_layers_count

                    )

                    dataset_grading.model_init()
                    dataset_grading.train()

                continue

                if df_name == "concatenated_domains" or df_name == "concatenated_datasets":

                    sampling_group = "dataset_name"
                    datasets = DATASETS
                    if df_name == "concatenated_domains":
                        sampling_group = "domain"
                        datasets = DOMAINS

                    for dataset_name_to_split in datasets:

                        if dataset_name_to_split in LEFT_OUT_DATASET_SKIP:

                            print(f"{dataset_name_to_split} inside left out dataset!")
                            continue
                            
                        print(f"\n\n*** start time: {datetime.now()} ***")
                        print(f"Running {model_name} on {df_name}")
                        
                        print(f"Running left out dataset: {dataset_name_to_split}")

                        # Now you can do whatever you want with the file
                        dataset = Dataset_Torch(
                            dir = base_dir,
                            file_name = df_name,
                            seed = SEED,
                            batch_size=128,
                            sample_size=2000,
                            sampling_group=sampling_group,
                            left_out_dataset=dataset_name_to_split
                        )
                        saved_model_dir = f"grading_models/BERT/saved_models/{model_name}/{description}/{dataset_name_to_split}/{dataset['name']}"

                        dataset.split_datasets()
                        dataset.init_dataloaders()

                        dataset_grading = Py_Torch(
                            # parent
                            model=BertForSequenceClassification.from_pretrained(model_name, num_labels=1),
                            dataset=dataset,
                            measurement_settings=Measurement_Settings(
                                dataset_name=dataset["name"],
                                embedding_seperated=False,
                                sentence_embedding_method=None,
                                feature_engenearing_method=None,

                                embedding_model_name=model_name,
                                grading_model=model_name,
                                
                                seed_data_split=42,
                                left_out_dataset=dataset_name_to_split,

                                description = description,

                                # inform user settings
                                print_regression=True,
                                print_classification=True,
                                
                                # save settings
                                settings_performance_tracking=ADD,
                                save_performance=True
                            ),

                            # child
                            y_column="assigned_points",

                            y_normalized=False, # idd not normalized because measurments are doen on non normalized values!

                            lr = 2e-5,
                            saved_model_dir = saved_model_dir,
                            epochs_to_run = 5,

                            unfrozen_layers_count=unfrozen_layers_count

                        )

                        dataset_grading.model_init()
                        dataset_grading.train()
