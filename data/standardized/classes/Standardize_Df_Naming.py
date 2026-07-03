
# libaries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# print
from services.printing.print_section import print_sub_section_start, print_sub_section_end

# constants
from data.standardized.constants import *

class Standardize_Df_Naming:

    def __init__(
            self,
            df,
            df_name,

            graders: None,
            domain_per_question: None,

            row_id,
            question,
            question_id,
            
            student_answer,
            reference_answer,
            assigned_points,
            
            max_points,
            domain,
        ):

            self.df = df
            self.df_name = df_name

            self.graders = graders
            self.domain_per_question = domain_per_question

            self.row_id = row_id
            self.question = question
            self.question_id = question_id
            
            self.student_answer = student_answer
            self.reference_answer = reference_answer
            self.assigned_points = assigned_points
            
            self.max_points = max_points
            self.domain = domain
    
    # run all standardadization opperations
    def standardize_df(self):
        
        print_sub_section_start(f"Standardizing: {self.df_name}")
        
        self.fix_missing_before()

        # standardize
        self.drop_useless_columns()
        self.standardize_column_names()
        self.full_missing_columns()

        self.fix_missing_after()

        # add name dataset
        self.df["dataset_name"] = self.df_name

        print_sub_section_end(f"Standardizing: {self.df_name}")
    
    # add missing values before standardization
    def fix_missing_before(self):

        # add row index to df if none existant
        if self.row_id.column == False and self.row_id.value == None:
            self.df = self.df.reset_index()

            # indicate that index has a column now
            self.row_id.name = "index"
            self.row_id.column = True

        # check if there are multiple graders
        if self.graders != None:

            # get average of grades
            self.multiple_graders()

    # add missing values after standardization
    def fix_missing_after(self):

        # add domains from list if necissary
        if self.domain_per_question != None:
            
            self.add_domains()

        # add max points based on assigned points
        if self.max_points.column == False and self.max_points.value == None:
            
            self.add_max_points()

        # add reference answers
        if self.reference_answer.column == False and self.reference_answer.value == None:

            self.add_reference_answer()

    # opperations
    def drop_useless_columns(self):
        
        # get class defined columns
        class_vars = self.get_class_variables()

        # get relevent column names
        relevant_column_names = []
        for key, value in class_vars.items():
            
            # check if column exists in df
            if value.column == True:
                relevant_column_names.append(value.name)
        
        # drop irrelivent columns
        self.df = self.df.loc[:, relevant_column_names]

    def standardize_column_names(self):
         
        # get class defined columns
        class_vars = self.get_class_variables()

        # dict for og name: new name
        rename_columns = {}

        for key, value in class_vars.items():
            
            # check if column exists in df
            if value.column == True:
                
                # the initial column name of the df is the value of the standard column name of the class
                rename_columns[value.name] = key

        # rename columns
        self.df = self.df.rename(columns=rename_columns)

    def full_missing_columns(self):

        # get class defined columns
        class_vars = self.get_class_variables()

        # add dummy variable to empty columns
        add_columns = [{"name": key, "value": value.value} for (key, value) in class_vars.items() if value.column == False]
        add_columns = {data["name"]: data["value"] for data in add_columns}
        self.df = self.df.assign(**add_columns)

    # services
    def get_class_variables(self):

        # get class defined columns
        class_vars = vars(self).copy()
        del class_vars["df"]
        del class_vars["df_name"]
        del class_vars["graders"]
        del class_vars["domain_per_question"]

        return class_vars

    # multiple graders
    def multiple_graders(self):

        # add all graders grades up
        for prade_column in self.graders:

            # define var
            if self.graders[0] == prade_column:
                total = self.df[prade_column]
            
            # add additional graders
            else:
                total += self.df[prade_column]

        # get avg
        self.df["assigned_points"] = total / len(self.graders)

    # add domains based on list
    def add_domains(self):

        # for every row in ASAP_sas_with_ref add domain
        for index, row in self.df.iterrows():
            
            # get domain indexed on question_id
            row_domain = self.domain_per_question[row["question_id"]-1]
            
            # add domain to df
            self.df.loc[index, "domain"] = row_domain

    # add max points based on assigned points
    def add_max_points(self):
        # get max points per score
        df_max_points = pd.DataFrame()
        df_max_points["max_points"] = self.df.groupby("question_id")["assigned_points"].max()

        # drop max points from df for merge
        self.df = self.df.drop(["max_points"], axis=1)

        # add max max_points column for every qeustion to ASAP_sas
        self.df = self.df.merge(df_max_points, on='question_id', how="left")

    # add reference answer based on student answers
    def add_reference_answer(self):

        # get all student answers with full points
        df_full_points = self.df.query("assigned_points == max_points")

        # Preprocessing
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()

        # quick pre-processing
        def preprocess(text):
            # Remove punctuation and lowercase
            text = "".join([char.lower() for char in text if char.isalpha() or char.isspace()])
            # Remove stop words
            text = " ".join([word for word in text.split() if word not in stop_words])
            # Stem words
            text = " ".join([ps.stem(word) for word in text.split()])
            return text

        # pre-process answers
        df_full_points["student_answer_processed"] = df_full_points["student_answer"].apply(preprocess)

        # make sure indexes align
        df_full_points = df_full_points.reset_index(drop=True)
        self.df = self.df.reset_index(drop=True)

        # Convert all student answers to vectors
        vectorizer = TfidfVectorizer()
        answer_vectors = vectorizer.fit_transform(df_full_points["student_answer_processed"])

        # group on question_id and get range of indexes for each group
        index_ranges = df_full_points.groupby('question_id').apply(lambda x: range(x.index.min(), x.index.max()+1))

        # save the index ranges for each question_id
        question_id_index = {
            "question_id": [],
            "min_i": [],
            "max_i": []
        }
        for category, index_range in index_ranges.items():
            question_id_index["question_id"].append(category)
            question_id_index["min_i"].append(list(index_range)[0])
            question_id_index["max_i"].append(list(index_range)[-1])

        question_id_index_df = pd.DataFrame(question_id_index)

        # get the average vector of all the vectors within this index range
        question_answer_vectors = {}

        # loop through all student answer vectors
        for index, vector in enumerate(answer_vectors):
            
            # get question_id of answer vector
            question_id = question_id_index_df.query("max_i >= @index & min_i <= @index")["question_id"].to_numpy()[0]
            
            # add answer vector
            if question_id in question_answer_vectors:
                question_answer_vectors[question_id].append(answer_vectors[index])
            else:
                question_answer_vectors[question_id] = [answer_vectors[index]]

        question_avg_answer_vectors = {}

        for key, value in question_answer_vectors.items():
            
            question_avg_answer_vectors[key] = (sum(value) / len(value))
        
        df_full_points["Q_A_similarity"] = None

        # for every row in df get the similarity score of the mean question vector
        for index, row in df_full_points.iterrows():
            
            # get student answer vector
            answer_vec = answer_vectors[index]
            
            # add similarity score (comparing answer vec with average answer vec for the related question) to answer ASAP_df
            df_full_points.loc[index, "Q_A_similarity"] = cosine_similarity(answer_vec, question_avg_answer_vectors[row["question_id"]])[0][0]
        
        # make Q_A_similarity float values
        df_full_points["Q_A_similarity"] = df_full_points["Q_A_similarity"].astype(float)

        max_rows = df_full_points.groupby('question_id').apply(lambda x: x.loc[x['Q_A_similarity'].idxmax()])

        # for every row in df add reference answer
        for index, row in self.df.iterrows():
            
            # get row with refrence answer
            reference_row = max_rows.query("question_id == @row['question_id']")
            
            # add similarity score (comparing answer vec with average answer vec for the related question) to answer ASAP_df
            self.df.loc[index, "reference_answer"] = reference_row["student_answer"].values[0]

    def reset_question_id(self, count):
        # reset former id to none every new dataset
        former_Q_Id_value = None

        # loop through all student answers
        for index, row in self.df.iterrows():
            current_Q_Id_value = row["question_id"]
            
            # check if question_id changed
            if former_Q_Id_value == current_Q_Id_value:
                self.df.loc[index, "question_id"] = count
            else:
                count = count + 1
                self.df.loc[index, "question_id"] = count
            
            former_Q_Id_value = current_Q_Id_value

        # ensure question id has type int
        self.df["question_id"] = self.df["question_id"].astype(int)

        return count

    def save_df(self):

        self.df.to_csv(f"{SAVE_STANDARDIZED}/{self.df_name}.csv", index=False)
