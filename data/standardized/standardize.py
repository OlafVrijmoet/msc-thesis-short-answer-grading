
# libaries
import pandas as pd

# classes
from data.standardized.classes.Standardize_Df_Naming import Standardize_Df_Naming
from data.standardized.classes.Relevant_data import Relevant_data

# services local
from data.standardized.services.retrieve_relevant_columns import retrieve_relevant_columns

# services
from services.import_csvs_from_dir import import_csvs_from_dir

# print
from services.printing.print_chapter import print_sub_chapter_start, print_sub_chapter_end
from services.printing.print_warning import print_warning

# constants
from data.standardized.constants import *

def standardize():

    print_sub_chapter_start(CHAPTER_1)

    datasets = {}

    import_csvs_from_dir(
        dict_datasets=datasets,
        dir_datasets=BASE_RAW
    )

    print_sub_chapter_end(CHAPTER_1)

    print_sub_chapter_start(CHAPTER_2)

    dataset_classes = {}

    # defining dataset setup classes
    for name, dataset in datasets.items():
        
        if name == "ASAP_sas":

            dataset_classes["ASAP_sas"] = Standardize_Df_Naming(
                df=dataset,
                df_name="ASAP_sas",
            
                graders=["Score1", "Score2"],
                domain_per_question=[
                    "science",
                    "science",
                    "english_language_arts",
                    "english_language_arts",
                    "biology",
                    "biology",
                    "english",
                    "english",
                    "english",
                    "science"
                ],

                row_id=Relevant_data(name="Id"),
                question=Relevant_data(name="question", column=False, value=None),
                question_id=Relevant_data(name="EssaySet"),
                
                student_answer=Relevant_data(name="EssayText"),
                reference_answer=Relevant_data(name="ref_answer", column=False, value=None),
                assigned_points=Relevant_data(name="assigned_points"),
                
                max_points=Relevant_data(name="max_points", column=False, value=None),
                domain=Relevant_data(name="domain", column=False, value=None),
            )

        elif name =="beetle":

            dataset_classes["beetle"] = Standardize_Df_Naming(
                df=dataset,
                df_name="beetle",

                graders=None,
                domain_per_question=None,

                row_id=Relevant_data(name="row_id"),
                question=Relevant_data(name="question"),
                question_id=Relevant_data(name="question_id"),
                
                student_answer=Relevant_data(name="student_answer"),
                reference_answer=Relevant_data(name="reference_answer"),
                assigned_points=Relevant_data(name="assigned_points"),
                
                max_points=Relevant_data(name="max_points"),
                domain=Relevant_data(name="domain"),
            )
    
        elif name == "nn_course":

            dataset_classes["neural_course"] = Standardize_Df_Naming(
                df=dataset,
                df_name="neural_course",

                graders=None,
                domain_per_question=None,

                row_id=Relevant_data(name="Unnamed: 0"),
                question=Relevant_data(name="question"),
                question_id=Relevant_data(name="question_id"),
                
                student_answer=Relevant_data(name="student_answer"),
                reference_answer=Relevant_data(name="ref_answer"),
                assigned_points=Relevant_data(name="grades_round"),
                
                max_points=Relevant_data(name="max_points", column=False, value=2),
                domain=Relevant_data(name="domain", column=False, value="neural_networks"),
            )

        elif name == "sciEntsBank":

            dataset_classes["sciEntsBank"] = Standardize_Df_Naming(
                df=dataset,
                df_name="sciEntsBank",

                graders=None,
                domain_per_question=None,

                row_id=Relevant_data(name="row_id"),
                question=Relevant_data(name="question"),
                question_id=Relevant_data(name="question_id"),
                
                student_answer=Relevant_data(name="student_answer"),
                reference_answer=Relevant_data(name="reference_answer"),
                assigned_points=Relevant_data(name="assigned_points"),
                
                max_points=Relevant_data(name="max_points"),
                domain=Relevant_data(name="domain"),
            )

        elif name == "Texas":

            dataset_classes["texas"] = Standardize_Df_Naming(
                df=dataset,
                df_name="texas",

                graders=None,
                domain_per_question=None,

                row_id=Relevant_data(name="index", column=False),
                question=Relevant_data(name="question"),
                question_id=Relevant_data(name="id"),
                
                student_answer=Relevant_data(name="student_answer"),
                reference_answer=Relevant_data(name="desired_answer"),
                assigned_points=Relevant_data(name="score_avg"),
                
                max_points=Relevant_data(name="max_points", column=False, value=5),
                domain=Relevant_data(name="domain", column=False, value="science"),
            )
        
        else:
            print_warning(f"{name} not found in {BASE_RAW}. If you added a dataset you have to create a custom standardize class inside ./data.standardize/standardize.py!")

    print_sub_chapter_end(CHAPTER_2)
    print_sub_chapter_start(CHAPTER_3)

    for name, dataset_class in dataset_classes.items():

        # standardize
        dataset_class.standardize_df()
    
    print_sub_chapter_end(CHAPTER_3)
    print_sub_chapter_start(CHAPTER_4)

    # resetting question id's
    count = -1
    for name, dataset_class in dataset_classes.items():

        count = dataset_class.reset_question_id(count)

    print_sub_chapter_end(CHAPTER_4)
    print_sub_chapter_start(CHAPTER_5)

    for name, dataset_class in dataset_classes.items():

        # save datasets
        dataset_class.save_df()


    print_sub_chapter_end(CHAPTER_5)
