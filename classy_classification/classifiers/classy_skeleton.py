from typing import List, Union

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, OneClassSVM
from spacy import util
from spacy.language import Language
from spacy.tokens import Doc, Span


class classySkeleton(object):
    def __init__(
        self,
        nlp: Language,
        name: str,
        data: dict,
        include_doc: bool,
        include_sent: bool,
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

    def __call__(self, text: str) -> dict:
        """predict the class for an input text

        Args:
            text (str): an input text

        Returns:
            dict: a key-class proba-value dict
        """
        embeddings = self.get_embeddings(text)
        embeddings = embeddings.reshape(1, -1)

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
        tuned_parameters = [{"C": C, "kernel": [str(k) for k in kernels]}]
        folds = self.config["max_cross_validation_folds"]
        cv_splits = max(2, min(folds, np.min(np.bincount(self.y)) // 5))
        if len(self.label_list):
            svm = SVC(C=1, probability=True, class_weight="balanced")
            self.clf = GridSearchCV(
                svm,
                param_grid=tuned_parameters,
                n_jobs=1,
                cv=cv_splits,
                scoring="f1_weighted",
                verbose=0,
            )
        else:
            svm = OneClassSVM(probability=True)
            self.clf = GridSearchCV(svm, param_grid=tuned_parameters, cv=cv_splits, n_jobs=1, verbose=0)

        self.clf.fit(self.X, self.y)

    def set_config(self, config):
        if config is None:
            config = {"C": [1, 2, 5, 10, 20, 100], "kernels": ["linear"], "max_cross_validation_folds": 5}
        self.config = config

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


class classySkeletonFewShotMultiLabel(classySkeleton):
    def set_config(self, config):
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
