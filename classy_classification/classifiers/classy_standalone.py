from typing import List, Union

from .classy_spacy import ClassyExternal, ClassySkeletonFewShot


class ClassyStandalone(ClassyExternal):
    def __call__(self, text: str) -> dict:
        """predict the class for an input text

        Args:
            text (str): an input text

        Returns:
            dict: a key-class proba-value dict
        """
        embeddings = self.get_embeddings([text])

        return self.get_prediction(embeddings)[0]

    def pipe(self, text: List[str]) -> List[dict]:
        """retrieve predicitons for multiple texts

        Args:
            text (List[str]): a list of texts

        Returns:
            List[dict]: list of key-class proba-value dict
        """
        embeddings = self.get_embeddings(text)

        return self.get_prediction(embeddings)


class ClassySentenceTransformer(ClassyStandalone, ClassySkeletonFewShot):
    def __init__(
        self,
        data: dict,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        multi_label: bool = False,
        config: Union[dict, None] = None,
        verbose: bool = False,
    ) -> None:
        """initialize a classy skeleton for classification using a SVC config and some input training data.

        Args:
            data (dict): training data. example
                {
                    "class_1": ["example"],
                    "class 2": ["example"]
                },
            device (str): device "cuda"/"cpu",
            config (dict, optional): a SVC config.
                example
                {
                    "C": [1, 2, 5, 10, 20, 100],
                    "kernel": ["linear"],
                    "max_cross_validation_folds": 5
                }.
        """
        self.multi_label = multi_label
        self.data = data
        self.model = model
        self.device = device
        self.verbose = verbose
        self.set_embedding_model()
        self.set_training_data()
        self.set_config(config)
        self.set_classification_model()
