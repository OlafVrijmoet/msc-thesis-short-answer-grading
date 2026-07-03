
import os
from tqdm import tqdm

import time
import re

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

import numpy as np

# services
from services.save import save

# classes
from classes.Grading_Model import Grading_Model

# constants
from performance_tracking.constants import ALL, TRAIN, TEST, VALIDATION, FALSE_PREDICTION

class Openai_Grading(Grading_Model):

  def __init__(self,
    # parent
    model, dataset, measurement_settings,

    # child
    y_column,

    y_normalized,

    shots

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
    super().__init__(model, dataset, measurement_settings, y_column, y_normalized, shots)

    # saving predictions
    self.y_pred = []

    # make y_pred of None's of length validation set - maybe even in df or np array!?

  def validation(self):
    """
    Perform validation for the regression grading model.
    """

    # measure performance on validation datasetsplit
    self.measure_performance(
        dataset_split=VALIDATION,
    )

  def make_predictions(self, dataset_split):

    # make sure it starts from the index given
    start_index = self.performance_tracking[dataset_split]["last_pred_index"]
    
    print(f"Running api calls for following dataset: {self.dataset['name']}")

    print("dataset_split", dataset_split)
    print("start_index", start_index)
    
    # Loop through the sampled dataframe from the start_index
    for index, row in tqdm(self.dataset[dataset_split].iloc[start_index:].iterrows(), total=self.dataset[dataset_split].iloc[start_index:].shape[0]):
        
        # Get the predicted points using the grade_student_answer function
        predicted_points = self.grade_student_answer(row=row, model=self.model, dataset_split=dataset_split, index=index)

        # save predicted_points in self.y_pred at index row
        self.y_pred.append(predicted_points)

    # return self.y_pred
    return self.y_pred

  def grade_student_answer(self, row, model, dataset_split, index):

    # Parameters for exponential backoff
    X = 5
    k = 2
    max_attempts = 5

    instruction_line = "Grade the following Student answer based on the Reference answer. The grade should be a howl number."

    messages = [
        {
            "role": "system",
            "content": "You are an AI trained to grade student answers based on a reference answer. Please return a single howl number."
        },
    ]

    for i in range(1, self.shots + 1):
        messages.append({
            "role": "system",
            "content": f"""
                {instruction_line}
                Student answer: {row[f'student_answer_{i}']}\n
                Reference answer: {row[f'reference_answer_{i}']}\n
                Grade out of {row['max_points']}: {row[f'assigned_points_{i}']}\n\n
            """
        })

    messages.append({
        "role": "system",
        "content": f"""
            {instruction_line}
            Student answer: {row['student_answer']}\n
            Reference answer: {row['reference_answer']}\n
            Grade out of {row['max_points']}: 
        """
    })

    for attempt in range(max_attempts):
      try:
        response = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=3, n=1, stop=None, temperature=0.5)
        
        # If the API call is successful, we break the loop and don't retry
        break
      except Exception as e:

        print(f"Error on attempt {attempt + 1}: {str(e)}")
        
        # If we've reached max attempts, re-raise the exception
        if attempt + 1 == max_attempts:

          self.performance_tracking[VALIDATION].save()
          
          raise
        
        else:
            # Sleep before next attempt
            time.sleep(X + (attempt ** k))
  
    content = response.choices[0].message['content'].strip()

    # This regex pattern finds float numbers in a string
    float_number_pattern = r"[-+]?[0-9]*\.?[0-9]+"
    numbers = re.findall(float_number_pattern, content)
    
    if numbers:
      predicted_points = float(numbers[0])
    else:
      print(f"\nNot valid input!: {content}\n")
      print(f"Sent messages:\n: {messages}\n")
      predicted_points = FALSE_PREDICTION

    return predicted_points
