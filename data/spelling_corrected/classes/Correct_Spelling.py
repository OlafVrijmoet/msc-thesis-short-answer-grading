
# parant classes
from classes.Dataset import Dataset

class Correct_Spelling(Dataset):

    def __init__(self, df_name, model_name, datasets, language, columns_to_add=..., save_new_colums_as_torch=False) -> None:
        super().__init__(df_name, model_name, datasets, language, columns_to_add, save_new_colums_as_torch)

    def process_row(self, row):

        row_dict = super().process_row(row)
    
        # add gensim version of row
        row_dict[self.model_name] = row_dict["row"].copy()

        if self.datasets[self.model_name]["may_run_now"] == True and self.datasets[self.model_name]["done"] == False:

            # correct spelleing
            row_dict[self.model_name]["student_answer"] = self.correctSpelling(row_dict["row"]["student_answer"])
            row_dict[self.model_name]["reference_answer"] = self.correctSpelling(row_dict["row"]["reference_answer"])
        
        return row_dict
