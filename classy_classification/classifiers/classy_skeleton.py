from typing import List, Union

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class classySkeleton(object):
    def __init__(
        self,
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
        if config:
            self.config = config
        else:
            self.config = {"C": [1, 2, 5, 10, 20, 100], "kernels": ["linear"], "max_cross_validation_folds": 5}
        self.data = data

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

    def get_embeddings(self, _):
        """is overwritten by a superclass to retrieve embeddings"""
        pass

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
        for key, value in self.data.items():
            labels += len(value) * [key]
            X += value
        self.y = self.le.fit_transform(labels)
        self.X = self.get_embeddings(X)

        if data:  # update if overwritten
            self.set_svc()

    def set_svc(self, config: dict = None):
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
        self.clf = GridSearchCV(
            SVC(C=1, probability=True, class_weight="balanced"),
            param_grid=tuned_parameters,
            n_jobs=1,
            cv=cv_splits,
            scoring="f1_weighted",
            verbose=0,
        )
        self.clf.fit(self.X, self.y)
