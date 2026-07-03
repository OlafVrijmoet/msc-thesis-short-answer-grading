
from sklearn.model_selection import train_test_split

class Dataset_Settings:

    def __init__(self, 
                 
            # info
            df, 
            df_name,
            base_dir,

            # stettings 
            may_run_now, required, done = False, parquet=False, name_required_dataset=None, force_run=False,
            
            # delete
            train_df=None, test_df=None, validation_df=None,
            x_train=None, x_test=None, x_validation=None, y_train=None, y_test=None, y_validation=None

        ) -> None:
        
        # info
        self.df = df
        self.df_name = df_name
        self.base_dir = base_dir
        self.save_location = f"{base_dir}/{df_name}/data"
        
        # settings
        self.may_run_now = may_run_now
        self.required = required
        self.done = done # will check if it is done inside the Dataset class
        self.parquet = parquet
        self.force_run = force_run

        self.name_required_dataset = name_required_dataset

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
