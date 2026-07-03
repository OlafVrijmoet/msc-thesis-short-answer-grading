
# libaries
from ast import literal_eval
import numpy as np

# classes local
from classes.Dataset import Dataset

class Embed_Words(Dataset):

    def __init__(self, df_name, model_name, datasets, language, embedding_model) -> None:
        super().__init__(df_name, model_name, datasets, language)

        self.embedding_model = embedding_model

    def process_row(self, row):

        # include all processing from the apprent class function
        row_dict = super().process_row(row)

        row_dict[self.model_name] = row_dict["row"]

        if self.datasets[self.model_name]["may_run_now"] == True and self.datasets[self.model_name]["done"] == False:
            
            # embed words
            row_dict[self.model_name]["student_answer"] = self.create_answer_embeddings(row_dict[self.model_name]["student_answer"])
            row_dict[self.model_name]["reference_answer"] = self.create_answer_embeddings(row_dict[self.model_name]["reference_answer"])

        return row_dict

    def create_answer_embeddings(self, answer_text):
        
        words = literal_eval(answer_text)
        answer_vector = np.zeros(self.embedding_model.model.vector_size)
        for word in words:
            # if word in embeddings:
            #     answer_vector += embeddings.get_vector(f"/c/af/{word}")

            try:
                answer_vector += self.embedding_model.model[f"{self.embedding_model.dir_in_model_embedding}{word}"]
            except:
                answer_vector += np.zeros((self.embedding_model.model.vector_size,), dtype=np.float32)
                
        return answer_vector
