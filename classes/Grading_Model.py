
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# classes
from performance_tracking.classes.Performance_Row import Performance_Row

# constants
from performance_tracking.constants import ALL, TRAIN, TEST, VALIDATION, FALSE_PREDICTION, FILL_PREDICTIONS
from performance_tracking.constants import NO_SAVING, NO_PROMPT_NO_REPEAT

class Grading_Model:

    def __init__(self, model, dataset, measurement_settings, y_column, y_normalized, shots=0):
        """
        Initialize the Grading_Model class.

        Parameters:
        - model: object
            The model object.
        - dataset: Dataset
            The dataset object.
        - experiment_identification: Measurement_Settings
          The model Measurement_Settings and contains information already known to create an identifiable experiment in the measurements.
        """
        self.model = model
        self.dataset = dataset
        self.measurement_settings = measurement_settings
        self.y_column = y_column

        self.y_normalized = y_normalized

        self.shots = shots
        self.epochs = 0

        self.performance_tracking = {
            TRAIN: Performance_Row(
                
                # id what is being tracked
                dataset_name=self.measurement_settings.dataset_name,
                embedding_seperated=self.measurement_settings.embedding_seperated,
                embedding_model_name = self.measurement_settings.embedding_model_name,
                sentence_embedding_method = self.measurement_settings.sentence_embedding_method,
                feature_engenearing_method = self.measurement_settings.feature_engenearing_method,
                grading_model = self.measurement_settings.grading_model,
                seed_data_split = self.measurement_settings.seed_data_split,
                left_out_dataset = self.measurement_settings.left_out_dataset,
                description = self.measurement_settings.description,
                shots = shots,

                length_df = len(dataset[TRAIN]),
                y_true = dataset[TRAIN]["assigned_points"],

                dataset_split=TRAIN,

                # duplicates handeling
                settings_performance_tracking=self.measurement_settings.settings_performance_tracking
            ),
            TEST: Performance_Row(
                
                # id what is being tracked
                dataset_name=self.measurement_settings.dataset_name,
                embedding_seperated=self.measurement_settings.embedding_seperated,
                embedding_model_name = self.measurement_settings.embedding_model_name,
                sentence_embedding_method = self.measurement_settings.sentence_embedding_method,
                feature_engenearing_method = self.measurement_settings.feature_engenearing_method,
                grading_model = self.measurement_settings.grading_model,
                seed_data_split = self.measurement_settings.seed_data_split,
                left_out_dataset = self.measurement_settings.left_out_dataset,
                description = self.measurement_settings.description,
                shots = shots,

                length_df = len(dataset[TEST]),
                y_true = dataset[TEST]["assigned_points"],

                dataset_split=TEST,

                # duplicates handeling
                settings_performance_tracking=self.measurement_settings.settings_performance_tracking
            ),
            VALIDATION: Performance_Row(
                
                # id what is being tracked
                dataset_name=self.measurement_settings.dataset_name,
                embedding_seperated=self.measurement_settings.embedding_seperated,
                embedding_model_name = self.measurement_settings.embedding_model_name,
                sentence_embedding_method = self.measurement_settings.sentence_embedding_method,
                feature_engenearing_method = self.measurement_settings.feature_engenearing_method,
                grading_model = self.measurement_settings.grading_model,
                seed_data_split = self.measurement_settings.seed_data_split,
                left_out_dataset = self.measurement_settings.left_out_dataset,
                description = self.measurement_settings.description,
                shots = shots,

                length_df = len(dataset[VALIDATION]),
                y_true = dataset[VALIDATION]["assigned_points"],

                dataset_split=VALIDATION,

                # duplicates handeling
                settings_performance_tracking=self.measurement_settings.settings_performance_tracking
            )
        }

    def train(self):
        """
        *** customize ***
        Train the grading model.
        """
        # measure performance
        self.measure_performance(
            dataset_split=TRAIN,
        )

    def test(self):
        """
        *** customize ***
        Test the grading model.
        """

        # measure performance on test datasetsplit
        self.measure_performance(
            dataset_split=TEST,
        )

    def validation(self):
        """
        *** customize ***
        Perform validation for the grading model.
        """

        # measure performance on validation datasetsplit
        self.measure_performance(
            dataset_split=VALIDATION,
        )

    def measure_performance(self, dataset_split):
        """
        Measure the performance of the grading model on a dataset split.

        Parameters:
        - dataset_split: str
          The dataset split to measure performance on.
        """

        # ***

        # don't run if you don't want to save it (for example to prevent repeat) - quick fix
        
        # check if experiement is done before
        experiement_done_before = self.performance_tracking[dataset_split].check_for_duplicates()
        
        # to be able to re-use settings in class and run user selected settings with one var
        running_settings = self.performance_tracking[dataset_split].settings_performance_tracking
        # user responds no saving or general settings no saving, than skip it!
        if running_settings == NO_SAVING or (experiement_done_before == True and running_settings == NO_PROMPT_NO_REPEAT):
            print("Experiement will not be saved, so will not be done!")
            return None
        
        # ***

        # print model info settings ask to print performance
        if self.measurement_settings.print_regression == True or self.measurement_settings.print_classification == True:

            self.performance_tracking[dataset_split].print_experiement_info()

        # make predictions if print_regression, print_classification or save_performance are true
        if self.measurement_settings.print_regression == True or self.measurement_settings.print_classification == True or self.measurement_settings.save_performance == True:

            y_pred = self.make_predictions(dataset_split)

            y_pred_original = y_pred.copy()

            # update FALSE_PREDICTION to value 0
            y_pred = np.where(y_pred == FALSE_PREDICTION, 0.0, y_pred)
            y_pred = np.where(y_pred == FILL_PREDICTIONS, 0.0, y_pred)

            # scale prediction from normalized value back to the original points
            if self.y_normalized == True:
                y_pred = y_pred * self.dataset[dataset_split]["max_points"]

        if self.measurement_settings.print_regression == True or self.measurement_settings.save_performance == True:

            # measure regression accuracy
            self.mean_squared_error(dataset_split, y_pred)

            # print accuracy regression
            if self.measurement_settings.print_regression == True:

                self.performance_tracking[dataset_split].print_regression_preformance()

        if self.measurement_settings.print_classification == True or self.measurement_settings.save_performance == True:
            
            # measure classification performance
            self.classification_performance(dataset_split, y_pred)

            if self.measurement_settings.print_classification == True:
                
                self.performance_tracking[dataset_split].print_classification_performance()

        if self.measurement_settings.save_performance == True:
            
            # save predictions inside row class
            self.performance_tracking[dataset_split]["y_pred"] = y_pred_original
                
            # run saving
            self.performance_tracking[dataset_split].save()

    def make_predictions(self, dataset_split):
        """
        *** customize ***
        Make predictions using the grading model.
        """

        raise ValueError("No custome make_predictions fuction defined in the child class")

    def mean_squared_error(self, dataset_split, y_pred):
        """
        Calculate the mean squared error (MSE) between true and predicted values.

        Parameters:
        - performance_tracking: Performance_Row
            the row data for performance tracking
        - y_true: array-like
          The true values.
        - y_pred: array-like
          The predicted values.
        - avg_max_points: float
            average max points values in dataset

        Returns:
        - float
          The calculated mean squared error.
        """

        # get ground truth
        y_true = self.dataset[dataset_split][self.y_column]

        # get average max points
        avg_max_points = self.dataset[dataset_split]["max_points"].mean()

        # Pearson's correlation
        correlation, p_value = pearsonr(y_true, y_pred)
        self.performance_tracking[dataset_split]['pears_correlation'] = correlation
        self.performance_tracking[dataset_split]['p_value'] = p_value

        #!!!!!! sqrt means no mead for ** 2 of avg_max_points !!!!!!!!
        # FOR MSE: rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * (avg_max_points ** 2)

        # y_pred already normalized
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # save rmse for experiement
        self.performance_tracking[dataset_split]['rmse'] = rmse

    def classification_performance(self, dataset_split, y_pred):
        """
        Evaluate the classification performance based on true and predicted labels.

        Parameters:
        - performance_tracking: Performance_Row
            the row data for performance tracking
        - y_pred: array-like
          The predicted labels.

        Returns:
        - dict
          A dictionary containing various classification performance metrics.
        """

        # add predictions to the df as a column
        self.dataset[dataset_split]["y_pred"] = y_pred

        # round to closest round number and convert from float to int
        self.dataset[dataset_split]["pred_points"] = self.dataset[dataset_split]["y_pred"].round()
        self.dataset[dataset_split]["pred_points"] = self.dataset[dataset_split]["pred_points"].astype(int)

        # Calculate accuracy
        accuracy = accuracy_score(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"])

        # Calculate precision, recall, and F1-score
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"], average='macro')
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"], average='micro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"], average='weighted')

        self.performance_tracking[dataset_split]['accuracy'] = accuracy
        self.performance_tracking[dataset_split]['precision_macro'] = precision_macro
        self.performance_tracking[dataset_split]['recall_macro'] = recall_macro
        self.performance_tracking[dataset_split]['f1_macro'] = f1_macro
        self.performance_tracking[dataset_split]['precision_micro'] = precision_micro
        self.performance_tracking[dataset_split]['recall_micro'] = recall_micro
        self.performance_tracking[dataset_split]['f1_micro'] = f1_micro
        self.performance_tracking[dataset_split]['precision_weighted'] = precision_weighted
        self.performance_tracking[dataset_split]['recall_weighted'] = recall_weighted
        self.performance_tracking[dataset_split]['f1_weighted'] = f1_weighted
