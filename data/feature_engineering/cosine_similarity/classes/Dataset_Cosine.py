
# libaries
from scipy.spatial.distance import cosine

# classes
from classes.Dataset import Dataset

class Dataset_Cosine(Dataset):

    def process_row(self, row):

        # include all processing from the apprent class function
        row_dict = super().process_row(row)

        self.cosine_similarity(row_dict["row"]["student_answer"], row_dict["row"]["reference_answer"])

        return row_dict

    def cosine_similarity(self, vec1, vec2):

        # add value to array of the column to be added
        self.columns_to_add[self.model_name]["cosine_similarity"].append(1 - cosine(vec1, vec2))

        