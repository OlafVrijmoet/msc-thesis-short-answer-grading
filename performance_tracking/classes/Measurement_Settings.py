
class Measurement_Settings:

    def __init__(self,
                 
        # id'ing run
        dataset_name: str,
        embedding_seperated: bool,
        embedding_model_name: str,
        sentence_embedding_method: str,
        feature_engenearing_method: str,
        grading_model: str,
        seed_data_split: int,

        # inform user settings
        print_regression: bool,
        print_classification: bool,
        
        # save settings
        settings_performance_tracking: int,
        save_performance: bool,

        description: str = "",

        left_out_dataset = None,
    
    ) -> None:
        """
        Initialize the Measurement_Settings class.

        Parameters:
        - dataset_name: str
          The name of the dataset.
        - embedding_seperated: bool
          Indicator for whether embeddings are separated.
        - embedding_model_name: str
          The name of the embedding model.
        - sentence_embedding_method: str
          The sentence embedding method used.
        - feature_engenearing_method
          feature_engenearing: like cosine_similarity
        - grading_model: str
          The name of the grading model.
        - seed_data_split: int
          The seed for data splitting.

        - print_regression: bool
          Indicator for whether to print regression results.
        - print_classification: bool
          Indicator for whether to print classification results.

        - settings_performance_tracking: int
          The frequency of performance tracking.
        - save_performance: bool
          Indicator for whether to save performance.
        """
        
        # id'ing run
        self.dataset_name = dataset_name
        self.embedding_seperated = embedding_seperated
        self.embedding_model_name = embedding_model_name
        self.sentence_embedding_method = sentence_embedding_method
        self.feature_engenearing_method = feature_engenearing_method
        self.grading_model = grading_model
        self.seed_data_split = seed_data_split

        self.description = description

        # inform user settings
        self.print_regression = print_regression
        self.print_classification = print_classification
        
        # save settings
        self.settings_performance_tracking = settings_performance_tracking
        self.save_performance = save_performance

        self.left_out_dataset = left_out_dataset

    def __getitem__(self, key):
        return getattr(self, key)
