
class Process_Stages:

    def __init__(self, basic_processed) -> None:
        # part of basic_processed
        self.basic_processed = basic_processed
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def any_process_stages_true(self):
        return any(getattr(self, attr) for attr in self.__dict__)
