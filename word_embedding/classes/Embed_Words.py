
from ast import literal_eval
from tqdm import tqdm

# classes
from word_embedding.classes.Embed_Word_Params import Embed_Word_Params

# services
from services.save import save

# constants
from constants_dir.column_constants import *
from constants_dir.path_constants import DATA_STAGES

class Embed_Words:

    def __init__(self, name_df, name_model, df, model, embed_word):
        
        self.name_df = name_df
        self.name_model = name_model
        self.df = df

        self.model = model

        # custom function for model that embeds word
        self.embed_word = embed_word

    # get df
    
    
    def embed_df(self):

        embedded_reference_answers = []
        embedded_student_answers = []
        
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):

            # embed reference answer
            embedded_ref = self.embed_text(literal_eval(row[REFERENCE_ANSWER]))
            embedded_reference_answers.append(embedded_ref)

            # embed student answer
            embedded_ans = self.embed_text(literal_eval(row[STUDENT_ANSWER]))
            embedded_student_answers.append(embedded_ans)

        self.df["reference_answer"] = embedded_reference_answers
        self.df["student_answer"] = embedded_student_answers
    
    def embed_text(self, text):

        embedded_text = []

        for word in text:

            # embed word using own model and given embed function
            embedded_word = self.embed_word(Embed_Word_Params(self.model.model, word))

            # add embeded word to embedded text
            embedded_text.append(embedded_word)
        
        return embedded_text
    
    def save(self):

        # save raw
        save(
            dir=f"{DATA_STAGES}/embedded_by_{self.model.model_name}",
            file_name=self.name_df,
            df=self.df,
            parquet=True
        )
