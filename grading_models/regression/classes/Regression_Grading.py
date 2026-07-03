
import numpy as np

# classes
from classes.Grading_Model import Grading_Model

# constants
from performance_tracking.constants import ALL, TRAIN, TEST, VALIDATION

class Regression_Grading(Grading_Model):

    def __init__(self,
        # parent
        model, dataset, measurement_settings,

        # child
        x_column, y_column,

        y_normalized
        
    ):
        """
        Initialize the Regression_Grading class.

        Parameters:
        - model: Measurement_Settings
          The model Measurement_Settings and contains information already known to create an identifiable experiment in the measurements.
        - dataset: Dataset
          The dataset object.
        - trained: bool
            ensures that the model is not trained multiple times
        """
        super().__init__(model, dataset, measurement_settings, y_column, y_normalized)

        self.x_column=x_column
        
        self.trained = False

    def train(self):
        """
        Train the regression grading model.
        """
        
        # no need to fit regression multiple times on same training data
        if self.trained == False:
            # fit regression
            self.model = self.model().fit(self.dataset["train"][self.x_column].values.reshape(-1, 1), self.dataset["train"][self.y_column])
            self.trained = True

        print("\nTrained!\n")

        # Call the parent's train() function
        super().train()

    def test(self):
        """
        Test the regression grading model.
        """
        
        # measure performance on test datasetsplit
        self.measure_performance(
            dataset_split=TEST,
        )

    def validation(self):
        """
        Perform validation for the regression grading model.
        """

        # measure performance on validation datasetsplit
        self.measure_performance(
            dataset_split=VALIDATION,
        )

    def make_predictions(self, dataset_split):
        """
        Make predictions using the regression grading model.

        Parameters:
        - dataset_split: str
            The dataset split that the prediction is made on
        
        Returns:
        - array-like
          The predicted values.
        """
        # get predictions
        y_pred = self.model.predict(self.dataset[dataset_split][self.x_column].values.reshape(-1, 1))

        # to prevent errors if there are nan values in the predictions. !!! There might be something going wrong !!!
        y_pred = np.nan_to_num(y_pred, nan=0)

        return y_pred
    