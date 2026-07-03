
import os
from tqdm import tqdm

import pandas as pd

from transformers import AutoTokenizer

# services
from services.save import save
from services.get_df import get_df

# parant classes
from classes.Dataset import Dataset

class BERT_ASAG_Tokenization(Dataset):

    def __init__(self, df_name, model_name, PyTorch_pre_trained_model_name, datasets, language, columns_to_add, save_new_colums_as_torch=False) -> None:
        
        super().__init__(df_name, model_name, datasets, language, columns_to_add, save_new_colums_as_torch)

        self.tokenizer = AutoTokenizer.from_pretrained(PyTorch_pre_trained_model_name)

    def process_row(self, row):

        row_dict = super().process_row(row)
    
        # add gensim version of row
        row_dict[self.model_name] = row_dict["row"].copy()

        if self.datasets[self.model_name]["may_run_now"] == True and self.datasets[self.model_name]["done"] == False:

            if self.model_name == "BERT_tokens_spelling_corrected":
                
                # tokenize
                self.columns_to_add[self.model_name]["tokenized_for_BERT"].append(self.encode_sentence_pair(row_dict[self.model_name]["student_answer"], row_dict[self.model_name]["reference_answer"]))
            
            else:
                
                # tokenize
                self.columns_to_add[self.model_name]["tokenized_for_BERT"].append(self.encode_sentence_pair(row_dict["row"]["student_answer"], row_dict["row"]["reference_answer"]))

        return row_dict

    def encode_sentence_pair(self, student_answer, reference_answer, max_length=512):
        
        # Tokenize without truncation
        encoded_without_truncation = self.tokenizer.encode_plus(
            student_answer,
            text_pair=reference_answer,
            truncation=False,
            padding=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Get the length of the sequence without truncation
        seq_length_without_truncation = len(encoded_without_truncation['input_ids'][0])
        # Check if truncation has occurred
        if seq_length_without_truncation > max_length:
            print("\n**************")
            print(f"Text has been truncated: {len(student_answer)} {len(reference_answer)}")
            print(f"The truncated numbers: {seq_length_without_truncation} {max_length}")
            print("**************\n")

        encoded = self.tokenizer.encode_plus(
            student_answer,
            text_pair=reference_answer,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return encoded
