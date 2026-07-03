
from classes.Dataset import Dataset

class Dataset_Basic(Dataset):
    
    def process_row(self, row):
        
        # include all processing from the apprent class function
        row_dict = super().process_row(row)

        row_dict = {
            "basic_processed": row_dict["row"]
        }

        # check if basic_processed should run and only run if it is not done
        if self.datasets["basic_processed"]["may_run_now"] == True and self.datasets["basic_processed"]["done"] == False:
            # lower
            row_dict["basic_processed"]["student_answer"] = row_dict["basic_processed"]["student_answer"].lower()
            row_dict["basic_processed"]["reference_answer"] = row_dict["basic_processed"]["reference_answer"].lower()

            # remove non chars
            row_dict["basic_processed"]["student_answer"] = self.keep_only_text(row_dict["basic_processed"]["student_answer"])
            row_dict["basic_processed"]["reference_answer"] = self.keep_only_text(row_dict["basic_processed"]["reference_answer"])

            # remove extra whitespace
            row_dict["basic_processed"]["student_answer"] = self.strip_extra_whitespace(row_dict["basic_processed"]["student_answer"])
            row_dict["basic_processed"]["reference_answer"] = self.strip_extra_whitespace(row_dict["basic_processed"]["reference_answer"])

            # spelling check
            row_dict["basic_processed"]["student_answer"] = self.correctSpelling(row_dict["basic_processed"]["student_answer"])
            row_dict["basic_processed"]["reference_answer"] = self.correctSpelling(row_dict["basic_processed"]["reference_answer"])

        return row_dict
