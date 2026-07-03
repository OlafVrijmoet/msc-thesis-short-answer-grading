

import torch
from transformers import BertModel, BertTokenizer

# services
from services.string_array import array_to_str

# classes
from classes.Dataset import Dataset

class BERT_Embedding(Dataset):

    def __init__(self, 
            df_name, model_name, datasets, language,

            pre_trained_BERT_model_name, # 'bert-base-uncased'
                 
        ) -> None:
        super().__init__(df_name, model_name, datasets, language)

        # Initialize the tokenizer and the model
        self.tokenizer = BertTokenizer.from_pretrained(pre_trained_BERT_model_name)
        self.model = BertModel.from_pretrained(pre_trained_BERT_model_name)

    def process_row(self, row):

        # include all processing from the apprent class function
        row_dict = super().process_row(row)

        row_dict[self.model_name] = row_dict["row"].copy()

        if self.datasets[self.model_name]["may_run_now"] == True and self.datasets[self.model_name]["done"] == False:

            row_dict[self.model_name]["student_answer"] = self.tokenize_text(row_dict["row"]["student_answer"])
            row_dict[self.model_name]["reference_answer"] = self.tokenize_text(row_dict["row"]["reference_answer"])
        
        return row_dict

    def tokenize_text(self, text):

        # Tokenize the sentence and encode it to IDs
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # Create tensors
        input_ids = torch.tensor([input_ids])

        # Forward pass: compute the BERT embeddings
        with torch.no_grad():
            outputs = self.model(input_ids)

        # Extract the embeddings
        embeddings = outputs[0][0]

        return array_to_str(embeddings.numpy())
