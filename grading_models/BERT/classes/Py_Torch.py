
import os
import shutil
import torch

from sklearn.metrics import mean_squared_error

from transformers import AdamW, get_linear_schedule_with_warmup

# classes
from classes.Grading_Model import Grading_Model

# services
from services.save import save

class Py_Torch(Grading_Model):

    def __init__(self, 
        
        model, dataset, measurement_settings,

        lr,
        
        y_column, 
        
        y_normalized,

        saved_model_dir,

        epochs_to_run,
        
        shots=0,

        unfrozen_layers_count=None,

    ):
        
        super().__init__(model, dataset, measurement_settings, y_column, y_normalized, shots)

        self.optimizer = AdamW(model.parameters(), lr)

        self.scheduler = None
        self.total_steps = len(dataset["train"]) * epochs_to_run

        self.epochs_to_run = epochs_to_run
        self.starting_epoch = 0
        self.current_training_epoch = 0
        self.saved_model_dir = saved_model_dir

        self.unfrozen_layers_count = unfrozen_layers_count

        self.device = "mps" if getattr(torch,'has_mps',False) \
            else "gpu" if torch.cuda.is_available() else "cpu"
    
    def model_init(self):

        if self.epochs_to_run < 5:

            print("WARNING: under 5 epochs are not saved!!!")

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps)

        self.starting_epoch = self.get_latest_saved_model_epoch()

        if self.starting_epoch != 0:

            # Find the model in this directory and load it
            model_path = os.path.join(self.saved_model_dir, str(self.starting_epoch), "model.pth")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        else:

            print("No previously saved model found. Using initial model.")
        
        # Unfreeze all layers in the pre-trained model
        for param in self.model.parameters():
            param.requires_grad = True

        if self.unfrozen_layers_count is not None:

            print("\nFreezing layers\n")
        
            # freeze all layers in the pre-trained model
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze the top 1 layer(s)
            for i, param in enumerate(self.model.base_model.encoder.layer[-self.unfrozen_layers_count:].parameters()):
                param.requires_grad = True

        self.model.to(self.device)
    
    def saving_model(self):

        # get heighest epoch version of the model - first get latest saved epoch for deleted but delete after saving so not all is lost if error during saving
        latest_saved_epoch = self.get_latest_saved_model_epoch()

        # save model
        model_dir = os.path.join(self.saved_model_dir, str(self.get_current_training_epoch_for_model()))
        save(
            dir=model_dir,
            file_name = "model",
            df=self.model.state_dict(),
            file_type="pth"
        )
        
        # delete the older version unless the epoch number is % 5
        if (latest_saved_epoch) % 5 != 0:
            del_dir = os.path.join(self.saved_model_dir, str(latest_saved_epoch))
            if os.path.exists(del_dir):
                shutil.rmtree(del_dir)
            else:
                print("The directory does not exist")
    
    def train(self):

        # Training loop
        for epoch in range(self.epochs_to_run):
            
            self.current_training_epoch = self.get_current_training_epoch_for_model(epoch)
            print(f"Current run epoch: {epoch}")
            print(f"Current training epoch: {self.current_training_epoch}")

            self.performance_tracking["train"]["epochs"] = self.current_training_epoch
            self.performance_tracking["test"]["epochs"] = self.current_training_epoch
            self.performance_tracking["validation"]["epochs"] = self.current_training_epoch

            self.model.train()
            train_loss = 0

            # performance tracking
            predictions = []
            ground_truth = []
            num_correct_predictions = 0
            total_predictions = 0

            for batch in self.dataset["train_dataloader"]:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                normalized_points = batch["normalized_points"].to(self.device)
                max_points = batch["max_points"].to(self.device)
                assigned_points = batch["assigned_points"].to(self.device)

                self.model.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=normalized_points.unsqueeze(1))
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_loss += loss.item()

                # performance measurment
                num_correct_predictions, total_predictions = self.intermediat_metrics(
                    # grading
                    outputs=outputs, assigned_points=assigned_points, max_points=max_points,
                    
                    # for metrics
                    predictions=predictions,
                    ground_truth=ground_truth,

                    num_correct_predictions=num_correct_predictions,
                    total_predictions=total_predictions
                )

            self.saving_model()
            
            print(f"Epoch {epoch + 1}/{self.epochs_to_run} - Train loss: {train_loss / len(self.dataset['train'])}")

            print("\nTRAINING:")
            # print out performance metrics
            self.print_intermediat_metrics(epoch, ground_truth, predictions, num_correct_predictions, total_predictions)

            # Evaluate on test set every 3rd epoch
            if (self.current_training_epoch) % 5 == 0:
                
                self.performance_tracking["train"]["y_pred"] = predictions
                self.measure_performance(dataset_split="train")

                self.model.eval()
                # performance tracking
                predictions = []
                ground_truth = []
                num_correct_predictions = 0
                total_predictions = 0

                with torch.no_grad():

                    for batch in self.dataset["test_dataloader"]:

                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        normalized_points = batch["normalized_points"].to(self.device)
                        max_points = batch["max_points"].to(self.device)
                        assigned_points = batch["assigned_points"].to(self.device)

                        outputs = self.model(input_ids, attention_mask=attention_mask)

                        # performance measurment
                        num_correct_predictions, total_predictions = self.intermediat_metrics(
                            # grading
                            outputs=outputs, assigned_points=assigned_points, max_points=max_points,
                            
                            # for metrics
                            predictions=predictions,
                            ground_truth=ground_truth,

                            num_correct_predictions=num_correct_predictions,
                            total_predictions=total_predictions
                        )

                print("\nTEST:")
                # print out performance metrics
                self.print_intermediat_metrics(epoch, ground_truth, predictions, num_correct_predictions, total_predictions)

                self.performance_tracking["test"]["y_pred"] = predictions
                self.measure_performance(dataset_split="test")

                self.model.eval()
                # performance tracking
                predictions = []
                ground_truth = []
                num_correct_predictions = 0
                total_predictions = 0

                with torch.no_grad():

                    for batch in self.dataset["validation_dataloader"]:

                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        normalized_points = batch["normalized_points"].to(self.device)
                        max_points = batch["max_points"].to(self.device)
                        assigned_points = batch["assigned_points"].to(self.device)

                        outputs = self.model(input_ids, attention_mask=attention_mask)

                        # performance measurment
                        num_correct_predictions, total_predictions = self.intermediat_metrics(
                            # grading
                            outputs=outputs, assigned_points=assigned_points, max_points=max_points,
                            
                            # for metrics
                            predictions=predictions,
                            ground_truth=ground_truth,

                            num_correct_predictions=num_correct_predictions,
                            total_predictions=total_predictions
                        )

                print("\nVALIDATION:")

                # print out performance metrics
                self.print_intermediat_metrics(epoch, ground_truth, predictions, num_correct_predictions, total_predictions)

                self.performance_tracking["validation"]["y_pred"] = predictions
                self.measure_performance(dataset_split="validation")

            # Evaluate on test set every 3rd epoch
            if self.current_training_epoch % 3 == 0 and self.current_training_epoch % 5 != self.current_training_epoch % 3:

                self.model.eval()

                # performance tracking
                predictions = []
                ground_truth = []
                num_correct_predictions = 0
                total_predictions = 0

                with torch.no_grad():
                    for batch in self.dataset["test_dataloader"]:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        normalized_points = batch["normalized_points"].to(self.device)
                        max_points = batch["max_points"].to(self.device)
                        assigned_points = batch["assigned_points"].to(self.device)

                        outputs = self.model(input_ids, attention_mask=attention_mask)

                        # performance measurment
                        num_correct_predictions, total_predictions = self.intermediat_metrics(
                            # grading
                            outputs=outputs, assigned_points=assigned_points, max_points=max_points,
                            
                            # for metrics
                            predictions=predictions,
                            ground_truth=ground_truth,

                            num_correct_predictions=num_correct_predictions,
                            total_predictions=total_predictions
                        )

                print("\nTEST:")

                # print out performance metrics
                self.print_intermediat_metrics(epoch, ground_truth, predictions, num_correct_predictions, total_predictions)

    def get_current_training_epoch_for_model(self, epoch=None):

        if epoch == None:

            return self.performance_tracking["train"]["epochs"]

        return self.starting_epoch + epoch + 1

    def intermediat_metrics(self, 
            # grading
            outputs, assigned_points, max_points,
            
            # for metrics
            predictions,
            ground_truth,

            num_correct_predictions,
            total_predictions
        ):

        # performance measurment
        predicted_normalized_points = outputs.logits.squeeze().detach()
        predicted_points = predicted_normalized_points * max_points

        # performance measurement msqr
        predictions.extend(predicted_points.cpu().tolist())
        ground_truth.extend(assigned_points.cpu().tolist())

        # performance measurement accuracy
        rounded_predictions = torch.round(predicted_points)
        num_correct_predictions += torch.sum(rounded_predictions == assigned_points).item()
        total_predictions += assigned_points.size(0)

        return num_correct_predictions, total_predictions

    def print_intermediat_metrics(self, epoch, ground_truth, predictions, num_correct_predictions, total_predictions):

        # Calculate the mean squared error for training data
        train_mse = mean_squared_error(ground_truth, predictions)
        print(f"Train Mean Squared Error after {epoch + 1} epochs: {train_mse}")

        # Calculate the accuracy
        accuracy = num_correct_predictions / total_predictions
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def make_predictions(self, dataset_split):

        # return the y_pred for dataset_split from self.performance_tracking
        return self.performance_tracking[dataset_split]["y_pred"]

    def get_latest_saved_model_epoch(self):
        # Check if the directory exists
        if os.path.isdir(self.saved_model_dir):
            # Get a list of all completed epoch directories and sort them 
            epoch_dirs = sorted([int(d) for d in os.listdir(self.saved_model_dir) 
                                    if os.path.isdir(os.path.join(self.saved_model_dir, d)) and d.isdigit()])
            
            # Check if there are any directories in the list
            if epoch_dirs:
                # Get the last completed epoch
                return max(epoch_dirs)
            else:
                print("No saved epochs found in the model directory.")
                return 0
        else:
            print("Saved model directory does not exist.")
            return 0
