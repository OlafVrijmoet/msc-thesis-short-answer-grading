
from classes.Process_Stages import Process_Stages

class Process_Stages_Gensim(Process_Stages):

    def __init__(self, basic_processed, gensim) -> None:
        super().__init__(basic_processed)

        self.gensim = gensim
