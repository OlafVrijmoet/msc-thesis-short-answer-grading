
import numpy as np

def embed_text_gensim(params):

    try:
        # vector = params.model.get_vector(f"/c/af/{params.word}", norm=True)
        vector = params.model[f"/c/en/{params.word}"]

    except:
        vector = np.zeros((params.model.vector_size,), dtype=np.float32)
    
    return vector
