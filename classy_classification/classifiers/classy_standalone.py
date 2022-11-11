from typing import List, Union

from .classy_spacy import (
    classyExternal,
    classySkeletonFewShot,
    classySkeletonFewShotMultiLabel,
)


class classyStandalone(classyExternal):
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


class classySentenceTransformerFewShot(classyStandalone, classySkeletonFewShot):
    def __init__(
        self,
        model: str,
        device: str,
        data: dict,
        config: Union[dict, None] = None,
    ) -> None:
        """initialize a classy skeleton for classification using a SVC config and some input training data.

        Args:
            data (dict): training data. example
                {
                    "class_1": ["example"],
                    "class 2": ["example"]
                },
            device (str): device "cuda"/"cpu",
            config (_type_, optional): a SVC config.
                example
                {
                    "C": [1, 2, 5, 10, 20, 100],
                    "kernels": ["linear"],
                    "max_cross_validation_folds": 5
                }.
        """
        self.data = data
        self.model = model
        self.device = device
        self.set_config(config)
        self.set_embedding_model()
        self.set_training_data()
        self.set_classification_model()


class classySentenceTransformerMultiLabel(classyStandalone, classySkeletonFewShotMultiLabel):
    def __init__(
        self,
        model: str,
        device: str,
        data: dict,
        config: Union[dict, None] = None,
    ):
        self.data = data
        self.model = model
        self.device = device
        self.set_config(config)
        self.set_embedding_model()
        self.set_training_data()
        self.set_classification_model()

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


def classySentenceTransformer(
    data: dict,
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device: str = None,
    config: Union[dict, None] = None,
    multi_label: bool = False,
):
    if multi_label:
        return classySentenceTransformerMultiLabel(model, device, data, config)
    else:
        return classySentenceTransformerFewShot(model, device, data, config)
