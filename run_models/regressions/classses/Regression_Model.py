
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# classes
from performance_tracking.classes.Performance_Row import Performance_Row

# constants
from performance_tracking.constants import ALL, TRAIN, TEST, VALIDATION

class Regression_Model():

    def __init__(self, 
            
            # id what is being tracked
            embedding_seperated: bool,
            embedding_model_name, classfication_model_name, dataset_name,
            seed_data_split,

            # duplicates handeling
            settings_performance_tacking: int,
            measurement_settings,

            dataset, classification_model, x_column, y_column
        ) -> None:
        
        # naming
        self.embedding_seperated = embedding_seperated
        self.embedding_model_name = embedding_model_name
        self.classfication_model_name = classfication_model_name
        self.dataset_name = dataset_name
        self.seed_data_split = seed_data_split

        # settings
        self.settings_performance_tacking=settings_performance_tacking
        self.measurement_settings = measurement_settings

        self.dataset = dataset
        self.classification_model = classification_model

        # columns
        self.x_column = x_column
        self.y_column = y_column

        # track if already trained
        self.trained = False

    # train classification_model
    def train(self):
        
        # no need to fit regression multiple times on same training data
        if self.trained == False:
            # fit regression
            self.classification_model = self.classification_model().fit(self.dataset["train_df"][self.x_column].values.reshape(-1, 1), self.dataset["train_df"][self.y_column])
            self.trained = True

        # measure performance
        self.measure_performance(
            dataset_split=TRAIN,
        )
    
    # measure classification_model performance
    def test(self):

        # measure performance on test datasetsplit
        self.measure_performance(
            dataset_split=TEST,
        )
    
    def validation(self):

        # measure performance on validation datasetsplit
        self.measure_performance(
            dataset_split=VALIDATION,
        )

    def measure_performance(self, dataset_split):

        # create Performance_Row class
        performance_tracking = Performance_Row(
            
            # id what is being tracked
            embedding_seperated=True, # indicateds if two models are used
            embedding_model_name=self.embedding_model_name,
            classfication_model_name=self.classfication_model_name,
            dataset_name=self.dataset_name,
            dataset_split=dataset_split,
            seed_data_split=self.seed_data_split,

            # duplicates handeling
            settings_performance_tacking=self.settings_performance_tacking
        )

        # print model info settings ask to print performance
        if self.measurement_settings.print_regression == True or self.measurement_settings.print_classification == True:

            performance_tracking.print_experiement_info()

        if self.measurement_settings.print_regression == True or self.measurement_settings.save_performance == True:

            # measure regression accuracy
            performance_tracking = self.mean_squared_error(performance_tracking, dataset_split)

            # print accuracy regression
            if self.measurement_settings.print_regression == True:

                performance_tracking.print_regression_preformance()

        if self.measurement_settings.print_classification == True or self.measurement_settings.save_performance == True:
            
            # measure classification performance
            performance_tracking = self.classification_performance(performance_tracking, dataset_split)

            if self.measurement_settings.print_classification == True:
                
                performance_tracking.print_classification_performance()

        if self.measurement_settings.save_performance == True:

            # run saving
            performance_tracking.save()

    def make_predictions(self, dataset_split):

        # get predictions
        y_pred = self.classification_model.predict(self.dataset[dataset_split][self.x_column].values.reshape(-1, 1))
        
        # get average max points
        avg_max_points = self.dataset[dataset_split]["max_points"].mean()

        # get ground truth
        y_ground_truth = self.dataset[dataset_split][self.y_column]

        return y_pred, avg_max_points, y_ground_truth

    # calculates rmse and add it to self.performance
    def mean_squared_error(self, performance_tracking, dataset_split):
        
        # ***
            # multiply the mean squared error by the average max points to the power of two.
            # in this way we get the non normalized rmse
        # ***

        y_pred, avg_max_points, y_ground_truth = self.make_predictions(dataset_split)

        y_pred = np.nan_to_num(y_pred, nan=0)

        # Pearson's correlation
        correlation, p_value = pearsonr(y_ground_truth, y_pred)
        performance_tracking['pears_correlation'] = correlation
        performance_tracking['p_value'] = p_value

        #!!!!!! sqrt means no mead for ** 2 of avg_max_points !!!!!!!!
        # FOR MSE: rmse = np.sqrt(mean_squared_error(y_ground_truth, y_pred)) * (avg_max_points ** 2)

        # the error (distance between normalized pred and normalized actual value) is squared for rmse, so the avg_max_points also has to be squared to get the correct non-normalized rmse
        rmse = np.sqrt(mean_squared_error(y_ground_truth, y_pred)) * avg_max_points

        # save rmse for experiement
        performance_tracking['rmse'] = rmse

        # i think necissary
        return performance_tracking

    def classification_performance(self, performance_tracking, dataset_split):

        # get predictions
        y_pred = self.classification_model.predict(self.dataset[dataset_split][self.x_column].values.reshape(-1, 1))

        y_pred = np.nan_to_num(y_pred, nan=0)

        # add predictions to the df as a column
        self.dataset[dataset_split]["y_pred"] = y_pred

        # scale prediction from normalized value back to the original points
        self.dataset[dataset_split]["pred_points"] = self.dataset[dataset_split]["y_pred"] * self.dataset[dataset_split]["max_points"]

        # round to closest round number and convert from float to int
        self.dataset[dataset_split]["pred_points"] = self.dataset[dataset_split]["pred_points"].round()
        self.dataset[dataset_split]["pred_points"] = self.dataset[dataset_split]["pred_points"].astype(int)

        # Calculate accuracy
        accuracy = accuracy_score(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"])

        # Calculate precision, recall, and F1-score
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"], average='macro')
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"], average='micro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"], average='weighted')

        performance_tracking['accuracy'] = accuracy
        performance_tracking['precision_macro'] = precision_macro
        performance_tracking['recall_macro'] = recall_macro
        performance_tracking['f1_macro'] = f1_macro
        performance_tracking['precision_micro'] = precision_micro
        performance_tracking['recall_micro'] = recall_micro
        performance_tracking['f1_micro'] = f1_micro
        performance_tracking['precision_weighted'] = precision_weighted
        performance_tracking['recall_weighted'] = recall_weighted
        performance_tracking['f1_weighted'] = f1_weighted

        return performance_tracking

    def __getitem__(self, key):
        return getattr(self, key)
