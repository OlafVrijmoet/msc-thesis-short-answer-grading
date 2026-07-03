
import pandas as pd
import numpy as np
from datetime import datetime

# classes
from services.get_df import get_df
from services.save import save

# services
from services.prompt_user import prompt_user

# constants
from performance_tracking.constants import *

class Performance_Row:

    def __init__(self,
            # id what is being tracked
            dataset_name,
            embedding_seperated, # indicates if two models are used, one for embedding and one for classifying (True) or one model is used for embedding and classifying (False)
            embedding_model_name, 
            sentence_embedding_method,
            feature_engenearing_method,
            grading_model, # new parameter
            dataset_split, # is it training, test or validation - constants defined in performance_tracking
            seed_data_split,

            length_df,
            y_true,

            # duplicates handling
            settings_performance_tracking, # allows experiments with the same embedding_model_name, classification_model_name, dataset_name to be added to performance df without asking

            description="",

            y_pred = None,
            left_out_dataset=None,

            shots=0,
            epochs=0,

            # performance measurements, regression
            rmse=None,
            pears_correlation=None,
            p_value=None,

            # performance measurements, classification
            accuracy=None, 
            precision_macro=None, 
            recall_macro=None, 
            f1_macro=None,
            precision_micro=None, 
            recall_micro=None, 
            f1_micro=None,
            precision_weighted=None, 
            recall_weighted=None, 
            f1_weighted=None
        ) -> None:

        # dataframe of past performance, current experiment performance is added to this df
        self.past_performance = None

        # id'ing experiment
        self.row_id = 0  # default, if there is no other past performance
        self.dataset_name = dataset_name
        self.embedding_seperated = embedding_seperated
        self.embedding_model_name = embedding_model_name
        self.sentence_embedding_method = sentence_embedding_method
        self.feature_engenearing = feature_engenearing_method
        self.grading_model = grading_model
        self.dataset_split = dataset_split
        self.seed_data_split = seed_data_split
        self.left_out_dataset = left_out_dataset
        self.shots = shots
        self.epochs = epochs
        self.description = description

        self.finishing_previous_experiement = False
        self.finished_pred = True
        self.y_pred = np.full(length_df, np.nan) if y_pred == None else y_pred
        self.y_true = y_true
        self.last_pred_index = 0
        self.length_df = length_df
        self.past_pred_dict = f"performance_tracking/data/{dataset_name}"

        self.time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # duplicates handling
        self.settings_performance_tracking = settings_performance_tracking

        # performance measurements, regression
        self.rmse = rmse
        self.pears_correlation = pears_correlation
        self.p_value = p_value

        # performance measurements, classification
        self.accuracy = accuracy
        self.precision_macro = precision_macro
        self.recall_macro = recall_macro
        self.f1_macro = f1_macro
        self.precision_micro = precision_micro
        self.recall_micro = recall_micro
        self.f1_micro = f1_micro
        self.precision_weighted = precision_weighted
        self.recall_weighted = recall_weighted
        self.f1_weighted = f1_weighted

        # get the info of past run
        self.fetch_saved_performance()

    # severly alter!!!
    def save(self):
        
        # check if all predictions have been made
        if np.isnan(self.y_pred).any():

            # get the index of the first nan value
            self.last_pred_index = np.where(np.isnan(self.y_pred))[0][0]

        else:

            self.last_pred_index = self.length_df - 1
            self.finished_pred = True
            
        # fetch / create df for performance
        self.fetch_saved_performance()

        # check if experiement is done before
        experiement_done_before = self.check_for_duplicates()

        # to be able to re-use settings in class and run user selected settings with one var
        running_settings = self.settings_performance_tracking
        
        # check to prompt use, if indicated to prompt and experiement done before
        if experiement_done_before == True and running_settings == PROMPT_EXPERIMENT_DONE:

            # promt user for performance settings
            running_settings = prompt_user(
                prompt=f"""
                The experiment with embeddings: {self.embedding_model_name}, classifier: {self.grading_model}, dataset: {self.dataset_name}.
                Your options: \n
                    Replace results with oldest findings: {REPLACE} \n
                    Add results to dataframe as new result: {ADD} \n
                    Delete new findings: {NO_SAVING} \n
                """, 
                user_options_values={
                    REPLACE: REPLACE,
                    ADD: ADD,
                    NO_SAVING: NO_SAVING
                }
            )
        
        # user responds no saving or general settings no saving, than skip it!
        if running_settings == NO_SAVING or (experiement_done_before == True and running_settings == NO_PROMPT_NO_REPEAT):
            
            # stop function, don't save results of this experiment
            return None
        
        # create row with current experiement data
        row_data = {

            # id'ing experiment
            'row_id': self.row_id,
            'dataset_name': self.dataset_name,
            'embedding_seperated': self.embedding_seperated,
            'embedding_model_name': self.embedding_model_name,
            'sentence_embedding_method': self.sentence_embedding_method,
            'feature_engenearing': self.feature_engenearing,
            'grading_model': self.grading_model,
            'dataset_split': self.dataset_split,
            'seed_data_split': self.seed_data_split,
            'left_out_dataset': self.left_out_dataset,
            'shots': self.shots,
            'epochs': self.epochs,
            'description': self.description,
            
            "finished_pred": self.finished_pred,
            "last_pred_index": self.last_pred_index,

            'time_stamp': self.time_stamp,

            # performance measurements, regression
            'rmse': self.rmse,
            'pears_correlation': self.pears_correlation,
            'p_value': self.p_value,

            # performance measurements, classification
            'accuracy': self.accuracy,
            'precision_macro': self.precision_macro,
            'recall_macro': self.recall_macro,
            'f1_macro': self.f1_macro,
            'precision_micro': self.precision_micro,
            'recall_micro': self.recall_micro,
            'f1_micro': self.f1_micro,
            'precision_weighted': self.precision_weighted,
            'recall_weighted': self.recall_weighted,
            'f1_weighted': self.f1_weighted
        }

        if running_settings == ADD or experiement_done_before == False:

            # add row
            self.past_performance = self.past_performance.append(row_data, ignore_index=True)
        
        elif (experiement_done_before == True and running_settings == REPLACE):
            
            # Get the indices of the rows in the original DataFrame which match the conditions
            match_indices = self.past_performance[
                (self.past_performance['embedding_model_name'] == self.embedding_model_name) &
                (self.past_performance['grading_model'] == self.grading_model) &
                (self.past_performance['dataset_name'] == self.dataset_name) & 
                (self.past_performance['embedding_seperated'] == self.embedding_seperated) & 
                (self.past_performance['dataset_split'] == self.dataset_split) &
                (self.past_performance['seed_data_split'] == self.seed_data_split) &
                (self.past_performance['left_out_dataset'] == self.left_out_dataset) &
                (self.past_performance['sentence_embedding_method'] == self.sentence_embedding_method) &
                (self.past_performance['feature_engenearing'] == self.feature_engenearing) &
                (self.past_performance['shots'] == self.shots) &
                (self.past_performance['epochs'] == self.epochs) &
                (self.past_performance['description'] == self.description)
            ].index

            # Convert the time_stamp column to datetime format
            self.past_performance['time_stamp'] = pd.to_datetime(self.past_performance['time_stamp'])

            # Get the index of the row with the oldest time_stamp among the matched rows
            oldest_index = self.past_performance.loc[match_indices, 'time_stamp'].idxmin()

            # update the row
            self.past_performance.loc[oldest_index] = row_data

        else:
            print("Something weard happened. Results not saved!")
            print(f"experiement_done_before: {experiement_done_before}")
            print(f"running_settings: {running_settings}")

        # save df
        save(
            dir=DF_TRACKING_DIR,
            file_name=DF_TRACKING_FILE_NAME,
            df=self.past_performance
        )
        
        if self.dataset_split == VALIDATION:
            # save predictions
            self.save_past_predictions()
    
    # print info
    def print_experiement_info(self):

        print("\n-----")
        print(f"embedding_model_name: {self.embedding_model_name}")
        print(f"grading_model: {self.grading_model}")
        print(f"dataset_name: {self.dataset_name}")

    def print_regression_preformance(self):

        print("\n*** regression performance ***")
        print(f"rmse: {self.rmse}")
        print(f"pears_correlation: {self.pears_correlation}")
        print(f"p_value: {self.p_value}")

    def print_classification_performance(self):

        # just the weighted measurments
        print("\n*** classification performance ***")
        print(f"accuracy: {self.accuracy}")
        print(f"precision_weighted: {self.precision_weighted}")
        print(f"recall_weighted: {self.recall_weighted}")
        print(f"f1_weighted: {self.f1_weighted}")

    # --- Services ---
        # only used internally

    def get_past_predictions(self):

        found, df_name, past_predictions = get_df(dir=self.past_pred_dict, file_name=self.row_id)

        if found == False:

            return False
        
        # update y_pred with the 
        self.y_pred = past_predictions["y_pred"].values
    
    # save past_performance table
    def save_past_predictions(self):

        if self.left_out_dataset == None:

            found, df_name, past_predictions = get_df(dir=self.past_pred_dict, file_name=self.row_id)

            if found == False:
                
                # create pd with row_id as column with y_ped
                past_predictions = pd.DataFrame({
                    "y_pred": self.y_pred,
                    "y_true": self.y_true
                })

            else:

                # Check if lengths are the same
                if past_predictions.shape[0] != len(self.y_pred):
                    # Calculate how many elements need to be added
                    missing_elements = past_predictions.shape[0] - len(self.y_pred)
                    
                    # Extend self.y_pred with the appropriate number of 1000s
                    self.y_pred = np.append(self.y_pred, [FILL_PREDICTIONS] * missing_elements)

                # replace values of row_id with latest predictions
                past_predictions["y_pred"] = self.y_pred

            save(dir=self.past_pred_dict, file_name=self.row_id, df=past_predictions)

        else:

            found, df_name, past_predictions = get_df(dir=f"{self.past_pred_dict}/{self.left_out_dataset}", file_name=self.row_id)

            if found == False:
                
                # create pd with row_id as column with y_ped
                past_predictions = pd.DataFrame({
                    "y_pred": self.y_pred,
                    "y_true": self.y_true
                })

            else:

                # Check if lengths are the same
                if past_predictions.shape[0] != len(self.y_pred):
                    # Calculate how many elements need to be added
                    missing_elements = past_predictions.shape[0] - len(self.y_pred)
                    
                    # Extend self.y_pred with the appropriate number of 1000s
                    self.y_pred = np.append(self.y_pred, [FILL_PREDICTIONS] * missing_elements)

                # replace values of row_id with latest predictions
                past_predictions["y_pred"] = self.y_pred

            save(dir=f"{self.past_pred_dict}/{self.left_out_dataset}", file_name=self.row_id, df=past_predictions)

    # only do when saving, than the performance df will only be loaded into memory to add the row
    def fetch_saved_performance(self):
        found, df_name, past_performance = get_df(dir=DF_TRACKING_DIR, file_name=DF_TRACKING_FILE_NAME)
        
        if found == True:

            # save past performance df into past_performance df in class
            self.past_performance = past_performance

            if past_performance.empty:
                return False

            # find row with the lagest row_id value
            last_experiment_row = self.past_performance.loc[self.past_performance['row_id'].idxmax()]

            if last_experiment_row["finished_pred"] == False and self.dataset_split == VALIDATION:
                
                self.row_id = last_experiment_row["row_id"]
                self.last_pred_index = last_experiment_row["last_pred_index"]

                # !!! check if this is same experiement !!!
                if self.row_is_current_experiment(last_experiment_row) == False:

                    raise ValueError("This row is not the same as the last unfinished row! First, finish the other experiement before going to the next.")
                
                # take out row with row_id from self.past_performance
                self.past_performance = self.past_performance.query("row_id != @self.row_id")

                self.finishing_previous_experiement = True

                # get y_pred from tracked_performance df based on row_id
                self.get_past_predictions()

            else:
                # default is 0, so only change is len longer than 0
                if len(past_performance) != 0:
                    
                    # set row_id to max_value_id + 1
                    max_value_id = self.past_performance['row_id'].max()
                    self.row_id = max_value_id + 1

        else:

            # create new df with class column names
            column_names = [
                # id'ing experiment
                'row_id', 'embedding_seperated', 'embedding_model_name', 'sentence_embedding_method', 'feature_engenearing', 'grading_model', 'dataset_name', 'dataset_split', 'seed_data_split', 'left_out_dataset', 'description',
                'shots','epochs',

                "finished_pred",
                "last_pred_index",

                'time_stamp',
                
                # performance measurements, regression
                'rmse',
                'pears_correlation',
                'p_value',

                # performance measurements, classification
                'accuracy', 
                'precision_macro', 'recall_macro', 'f1_macro',
                'precision_micro', 'recall_micro', 'f1_micro',
                'precision_weighted', 'recall_weighted', 'f1_weighted'
            ]

            self.past_performance = pd.DataFrame(columns=column_names)

            # save new df - maybe don't do here?! - just when it's saved
            save(dir=DF_TRACKING_DIR, file_name=DF_TRACKING_FILE_NAME, df=self.past_performance)

    def row_is_current_experiment(self, row):

        if (row['embedding_model_name'] == self.embedding_model_name) and \
            (row['grading_model'] == self.grading_model) and \
            (row['dataset_name'] == self.dataset_name) and \
            (row['embedding_seperated'] == self.embedding_seperated) and \
            (row['dataset_split'] == self.dataset_split) and \
            (row['seed_data_split'] == self.seed_data_split) and \
            (row['left_out_dataset'] == self.left_out_dataset) and \
            (row['sentence_embedding_method'] == self.sentence_embedding_method) and \
            (row['feature_engenearing'] == self.feature_engenearing) and \
            (row['description'] == self.description) and \
            (row['shots'] == self.shots):
            return True
        else:
            return False

    # returns True if experiement done before
    def check_for_duplicates(self):

        if self.past_performance.empty:
            return False
        
        # gives df with all rows with unique experiement ids
        duplicate_row = self.past_performance[
            (self.past_performance['embedding_model_name'] == self.embedding_model_name) &
            (self.past_performance['grading_model'] == self.grading_model) &
            (self.past_performance['dataset_name'] == self.dataset_name) & 
            (self.past_performance['embedding_seperated'] == self.embedding_seperated) & 
            (self.past_performance['dataset_split'] == self.dataset_split) &
            (self.past_performance['seed_data_split'] == self.seed_data_split) &
            (self.past_performance['left_out_dataset'] == self.left_out_dataset) &
            (self.past_performance['sentence_embedding_method'] == self.sentence_embedding_method) &
            (self.past_performance['feature_engenearing'] == self.feature_engenearing) &
            (self.past_performance['shots'] == self.shots) &
            (self.past_performance['description'] == self.description) &
            (self.past_performance['epochs'] == self.epochs)
        ]

        # checks if df is empty, returns True if experiement done before
        return not duplicate_row.empty
    
    def current_row_id(self):

        found, _, past_performance = get_df(dir=DF_TRACKING_DIR, file_name=DF_TRACKING_FILE_NAME)
        
        if found == True:
            
            # default is 0, so only change is len longer than 0
            if len(past_performance) != 0:
                
                # set row_id to max_value_id + 1
                max_value_id = self.past_performance['row_id'].max()
                self.row_id = max_value_id + 1

        return self.row_id

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
