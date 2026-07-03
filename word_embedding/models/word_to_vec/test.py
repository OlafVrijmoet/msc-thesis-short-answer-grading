
#%%
import os
import pandas as pd
import gensim
from gensim.models import Word2Vec
import gensim.downloader as gensim_api
import ast

# %%

df = pd.read_csv("../../data/processed/data/lemmitized_data/domain/neural_networks.csv")

# %%
df

# %%
# Define a function to grade a student answer
def grade_answer(student_answer, reference_answers, model):

    # Calculate the average similarity between the student answer and the reference answers
    similarity_sum = 0
    for reference_answer in reference_answers:

        # Calculate the similarity between the student answer and the reference answer
        similarity = 0
        count = 0
        for word in student_answer:
            similarity += model.similarity(word, reference_answer)
            count += 1
        if count > 0:
            similarity_sum += similarity / count

    # Calculate the final score as the average similarity between the student answer and the reference answers
    score = similarity_sum / len(reference_answers)

    # Return the score as a percentage
    return round(score * 100)

# %%
# model = Word2Vec.load('word2vec.model')
grade_answer(ast.literal_eval(df.loc[0, "student_answer"]), ast.literal_eval(df.loc[0, "reference_answer"]), model)

# %%
df.loc[0, "assigned_points"]

#%%
df.loc[0, "max_points"]

# %%
words = list(model.wv.index_to_key)
words
# %%
words = list(model.wv.index_to_key)
words
# %%
rock_cnt = model.get_vector("rock", norm=True)
rock_cnt

# %%
# https://github.com/RaRe-Technologies/gensim-data
model = gensim_api.load("fasttext-wiki-news-subwords-300")


# %%
model.similarity("hello", "hey")
# %%
