import importlib.util
from typing import List, Union

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from spacy.language import Language
from spacy.tokens import Doc, Span

onnx = importlib.util.find_spec("fast-sentence-transformers")
if onnx is None:
    from sentence_transformers import SentenceTransformer
else:
    from fast_sentence_transformers import (
        FastSentenceTransformer as SentenceTransformer,
    )


class classySkeleton:
    def __init__(
        self,
        nlp: Language,
        name: str,
        data: dict,
        include_doc: bool = True,
        include_sent: bool = False,
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

        self.set_config(config)
        self.data = data
        self.name = name
        self.nlp = nlp
        self.include_doc = include_doc
        self.include_sent = include_sent
        if include_sent:
            Span.set_extension("cats", default=None, force=True)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        if include_doc:
            Doc.set_extension("cats", default=None, force=True)
        self.set_training_data()
        self.set_classification_model()

    def set_training_data(self):
        """Overwritten by super class"""
        raise NotImplementedError("Needs to be overwritten by superclass")

    def set_classification_model(self):
        """Overwritten by super class"""
        raise NotImplementedError("Needs to be overwritten by superclass")

    def get_embeddings(self):
        """Overwritten by super class"""
        raise NotImplementedError("Needs to be overwritten by superclass")

    def __call__(self):
        """Overwritten by super class"""
        raise NotImplementedError("Needs to be overwritten by superclass")

    def pipe(self):
        """Overwritten by super class"""
        raise NotImplementedError("Needs to be overwritten by superclass")

    def get_prediction(self, embeddings: List[List]) -> List[dict]:
        """get the predicitons for a list om embeddings

        Args:
            embeddings (List[List]): a list of text embeddings.

        Returns:
            List[dict]: list of key-class proba-value dict
        """
        pred_result = self.clf.predict_proba(embeddings)

        return self.proba_to_dict(pred_result)


class classySkeletonFewShot(classySkeleton):
    def set_config(self, config: Union[dict, None] = None):
        """
        > This function sets the config attribute of the class to the config parameter if the config parameter is not None,
        otherwise it sets the config attribute to a default value

        :param config: A dictionary of parameters to be used in the SVM
        :type config: Union[dict, None]
        """
        if config is None:
            config = {"C": [1, 2, 5, 10, 20, 100], "kernels": ["linear"], "max_cross_validation_folds": 5}
        self.config = config

    def set_classification_model(self, config: dict = None):
        """Set and fit the SVC model.

        Args:
            config (dict, optional): A config containing keys for SVC kernels, C, max_cross_validation_folds.
                Defaults to None if self.config needs to be used.
        """
        if config:  # update if overwritten
            self.config = config

        C = self.config["C"]
        kernels = self.config["kernels"]

        folds = self.config["max_cross_validation_folds"]
        cv_splits = max(2, min(folds, np.min(np.bincount(self.y)) // 5))
        if len(self.label_list) > 1:
            tuned_parameters = [{"C": C, "kernel": [str(k) for k in kernels]}]
            svm = SVC(C=1, probability=True, class_weight="balanced")
            self.clf = GridSearchCV(
                svm,
                param_grid=tuned_parameters,
                n_jobs=1,
                cv=cv_splits,
                scoring="f1_weighted",
                verbose=0,
            )
        elif len(self.label_list) == 1:
            raise NotImplementedError(
                "I have not managed to take an in-depth look into probabilistic predictions for single class"
                " classification yet. Feel free to provide your input on"
                " https://github.com/Pandora-Intelligence/classy-classification/issues/12."
            )
        else:
            raise ValueError("Provide input data with Dict[key, List].")

        self.clf.fit(self.X, self.y)

    def proba_to_dict(self, pred_results: List[List]) -> List[dict]:
        """converts probability prediciton to a formatted key-class proba-value list

        Args:
            pred_results (_List[List]): a list of prediction probabilities.

        Returns:
            List[dict]: list of key-class proba-value dict
        """

        pred_dict = []
        for pred in pred_results:
            pred_dict.append({label: value for label, value in zip(self.le.classes_, pred)})

        return pred_dict

    def set_training_data(self, data: dict = None):
        """_summary_

        Args:
            data (dict, optional): a dict containing category keys and lists ov example values.
                Defaults to None if self.data needs to be used.
        """
        if data:  # update if overwritten
            self.data = data

        self.le = preprocessing.LabelEncoder()
        labels = []
        X = []
        self.label_list = list(self.data.keys())
        assert len(list(self.label_list)) == len(
            set(self.label_list)
        ), "Do not provide duplicate labels for training data."
        for key, value in self.data.items():
            labels += len(value) * [key]
            X += value
        self.y = self.le.fit_transform(labels)
        self.X = self.get_embeddings(X)

        if data:  # update if overwritten
            self.set_classification_model()


class classySkeletonFewShotMultiLabel(classySkeleton):
    def set_config(self, config: Union[dict, None] = None):
        """
        > This function sets the config attribute of the class to the config parameter if the config parameter is not None,
        otherwise it sets the config attribute to a default value

        :param config: A dictionary of parameters to be used in the SVM
        :type config: Union[dict, None]
        """
        if config is None:
            config = {"hidden_layer_sizes": (64,), "seed": 42}
        self.config = config

    def set_classification_model(self, config: dict = None):
        """Set and fit the Multi-layer Perceptron (MLP) classifier.

        Args:
            config (dict, optional): A config for MLPClassifier: hidden_layer_sizes, seed.
        """
        if config:  # update if overwritten
            self.config = config

        hidden_layer_sizes = self.config["hidden_layer_sizes"]
        seed = self.config["seed"]
        self.clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=seed)
        self.clf.fit(self.X, self.y)

    def proba_to_dict(self, pred_results: List[List]) -> List[dict]:
        """
        > It takes a list of lists of probabilities and returns a list of dictionaries where each dictionary has the label as
        the key and the probability as the value

        :param pred_results: List[List]
        :type pred_results: List[List]
        :return: A list of dictionaries.
        """
        pred_dict = []
        for pred in pred_results:
            pred_dict.append({label: value for label, value in zip(self.data.keys(), pred)})

        return pred_dict

    def set_training_data(self, data: dict = None):
        """
        The function takes in a dictionary of data, and sets the training data for the model

        :param data: a dictionary of lists of strings. The keys are the labels, and the values are the samples
        :type data: dict
        """
        if data:  # update if overwritten
            self.data = data

        if data:  # update if overwritten
            self.set_classification_model()

        X = np.unique([sample for values in self.data.values() for sample in values])
        self.X = self.get_embeddings(X.tolist())
        self.y = [[1 if sample in values else 0 for values in self.data.values()] for sample in X]


class classyExternal:
    def get_embeddings(self, docs: Union[List[Doc], List[str]]) -> List[List[float]]:
        """retrieve embeddings from the SentenceTransformer model for a text or list of texts

        Args:
            X (List[str]): input texts

        Returns:
            List[List[float]]: output embeddings
        """
        # inputs = self.tokenizer(X, padding=True, truncation=True, max_length=512, return_tensors="pt")
        # ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}

        # return self.session.run(None, ort_inputs)[0]
        docs = list(docs)
        if isinstance(docs, list):
            if isinstance(docs[0], str):
                pass
            elif isinstance(docs[0], Doc):
                docs = [doc.text for doc in docs]
        else:
            raise ValueError("This should be a List")

        return self.encoder.encode(docs)

    def set_embedding_model(self, model: str = None, device: str = "cpu"):
        """set the embedding model based on a sentencetransformer model or path

        Args:
            model (str, optional): the model name. Defaults to self.model, if no model is provided.
        """
        if model:  # update if overwritten
            self.model = model
        if device:
            self.device = device

        if onnx is None:
            if self.device in ["gpu", "cuda", 0]:
                self.device = None  # If None, checks if a GPU can be used.
            else:
                self.device = "cpu"
            self.encoder = SentenceTransformer(self.model, device=self.device)
        else:
            if device in ["gpu", "cuda", 0]:
                self.encoder = SentenceTransformer(self.model, device=self.device, quantize=False)
            else:
                self.encoder = SentenceTransformer(self.model, device=self.device, quantize=True)

        if model:  # update if overwritten
            self.set_training_data()
            self.set_classification_model()
