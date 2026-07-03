
# relevant info class
class Relevant_data:
    def __init__(self, name, column=True, value=None):
        self.name = name
        
        # if info is present in a df column
        self.column = column
        
        # default value
        self.value = value
