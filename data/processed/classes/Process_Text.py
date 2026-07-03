
import os

# libaries
import nltk
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from tqdm import tqdm

# services
from services.save import save

# print
from services.printing.print_section import print_sub_section_start, print_sub_section_end

# constants
from data.processed.constants import *

class Process_Text:

    def __init__(self, df, name):
        
        self.df_raw = df

        self.df_stemmed = df.copy()
        self.df_lemmitized = df.copy()

        self.name = name

    def process_text(self):

        print_sub_section_start(f"Text processing: {self.name}")

        # replace None values in all text with empty string
        self.df_raw[["student_answer", "reference_answer", "question"]] = self.df_raw[["student_answer", "reference_answer", "question"]].fillna('')

        self.itter_rows()

        print_sub_section_end(f"Text processing: {self.name}")
        print_sub_section_start(f"Saving: {self.name}")

        self.save()

        print_sub_section_end(f"Saving: {self.name}")

    def itter_rows(self):

        stemmed_student_answers = []
        stemmed_reference_answers = []
        stemmed_questions = []

        lemmatized_student_answers = []
        lemmatized_reference_answers = []
        lemmatized_questions = []

        spell = SpellChecker()

        for index, row in tqdm(self.df_raw.iterrows(), total=self.df_raw.shape[0]):
            
            student_answer = row["student_answer"]
            reference_answer = row["reference_answer"]
            question = row["question"]

            # lower
            student_answer = student_answer.lower()
            reference_answer = reference_answer.lower()
            question = question.lower()

            # remove non chars
            student_answer = self.keep_only_text(student_answer)
            reference_answer = self.keep_only_text(reference_answer)
            question = self.keep_only_text(question)

            # remove extra whitespace
            student_answer = self.strip_extra_whitespace(student_answer)
            reference_answer = self.strip_extra_whitespace(reference_answer)
            question = self.strip_extra_whitespace(question)

            # remove puctuation
            student_answer = self.strip_punctuation(student_answer)
            reference_answer = self.strip_punctuation(reference_answer)
            question = self.strip_punctuation(question)

            # tokenize
            student_answer = self.tokenize_words(student_answer)
            reference_answer = self.tokenize_words(reference_answer)
            question = self.tokenize_words(question)

            # spelling check
            student_answer = self.correctSpelling(spell, student_answer)
            reference_answer = self.correctSpelling(spell, reference_answer)
            question = self.correctSpelling(spell, question)

            # stem
            stemmed_student_answer = self.stem(student_answer)
            stemmed_reference_answer = self.stem(reference_answer)
            stemmed_question = self.stem(question)

            # save to stemmed df
            stemmed_student_answers.append(stemmed_student_answer)
            stemmed_reference_answers.append(stemmed_reference_answer)
            stemmed_questions.append(stemmed_question)

            # lemitize
            lemmatized_student_answer = self.lemmatize(student_answer)
            lemmatized_reference_answer = self.lemmatize(reference_answer)
            lemmatized_question = self.lemmatize(question)

            # save to lemitized df
            lemmatized_student_answers.append(lemmatized_student_answer)
            lemmatized_reference_answers.append(lemmatized_reference_answer)
            lemmatized_questions.append(lemmatized_question)
        
        # save processed text into stemmed df
        self.df_stemmed["student_answer"] = stemmed_student_answers
        self.df_stemmed["reference_answer"] = stemmed_reference_answers
        self.df_stemmed["question"] = stemmed_questions

        # save processed text into lemmatized df
        self.df_lemmitized["student_answer"] = lemmatized_student_answers
        self.df_lemmitized["reference_answer"] = lemmatized_reference_answers
        self.df_lemmitized["question"] = lemmatized_questions

    # sevice functions

    def keep_only_text(self, text: str) -> str:

        # define the regular expression pattern
        pattern = r'[^\w\s]'

        return text.replace(pattern, '')

    def strip_extra_whitespace(self, s: str) -> str:
        return " ".join(s.split())
    
    def tokenize_words(self, text):

        tokens = nltk.tokenize.word_tokenize(text)
        return tokens

    def correctSpelling(self, spell, text):
        # Initialize the spell checker
        return [spell.correction(token) if spell.correction(token) is not None else token for token in text]

    def strip_punctuation(self, text: str) -> str:

        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        return text
    
    def stem(self, tokenized_text):

        stemmer = SnowballStemmer("english")

        stemmed_text = []
        # loop through words and get sem
        for word in tokenized_text:
            if word.isalpha():
                stemmed_word = stemmer.stem(word)
                
            else:
                stemmed_word = word
            
            stemmed_text.append(stemmed_word)

        return stemmed_text

    def lemmatize(self, tokenized_text):

        lemmatizer = WordNetLemmatizer()

        lemmatized_text = []

        # loop through words and get sem
        for word in tokenized_text:
            
            if word.isalpha():
                stemmed_word = lemmatizer.lemmatize(word)
                
            else:
                stemmed_word = word
            
            lemmatized_text.append(stemmed_word)

        return lemmatized_text

    def save(self):

        # save raw
        save(
            dir=DF_RAW,
            file_name=self.name,
            df=self.df_raw
        )

        # save stemmed
        save(
            dir=DF_STEMMED,
            file_name=self.name,
            df=self.df_stemmed
        )

        # save lemmitized
        save(
            dir=DF_LEMMITIZED,
            file_name=self.name,
            df=self.df_lemmitized
        )
