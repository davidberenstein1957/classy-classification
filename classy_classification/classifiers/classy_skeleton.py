import importlib.util
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, OneClassSVM
from spacy.language import Language
from spacy.tokens import Doc, Span

onnx = importlib.util.find_spec("fast_sentence_transformers") or importlib.util.find_spec("fast-sentence-transformers")
if onnx is None:
    from sentence_transformers import SentenceTransformer
else:
    from fast_sentence_transformers import (
        FastSentenceTransformer as SentenceTransformer,
    )


class ClassySkeleton:
    def __init__(
        self,
        nlp: Language,
        name: str,
        data: dict,
        include_doc: bool = True,
        include_sent: bool = False,
        multi_label: bool = False,
        config: Union[dict, None] = None,
        verbose: bool = True,
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
                    "kernel": ["linear"],
                    "max_cross_validation_folds": 5
                }.
        """
        self.multi_label = multi_label

        self.data = data
        self.name = name
        self.nlp = nlp
        self.verbose = verbose
        self.include_doc = include_doc
        self.include_sent = include_sent
        if include_sent:
            Span.set_extension("cats", default=None, force=True)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        if include_doc:
            Doc.set_extension("cats", default=None, force=True)
        self.set_training_data()
        self.set_config(config)
        self.set_classification_model()

    def set_training_data(self):
        """Overwritten by super class"""
        raise NotImplementedError("Needs to be overwritten by superclass")

    def set_config(self):
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
        if len(self.label_list) > 1:
            pred_result = self.clf.predict_proba(embeddings)
            pred_result = self.proba_to_dict(pred_result)
        else:
            pred_result = self.clf.predict(embeddings)
            label = self.label_list[0]
            pred_result = [
                {label: 1, f"not_{label}": 0} if pred == 1 else {label: 0, f"not_{label}": 1} for pred in pred_result
            ]
        return pred_result


class ClassySkeletonFewShot(ClassySkeleton):
    def set_config(self, config: Union[dict, None] = None):
        """
        > This function sets the config attribute of the class to the config parameter if the config parameter is not None,
        otherwise it sets the config attribute to a default value

        :param config: A dictionary of parameters to be used in the SVM
        :type config: Union[dict, None]
        """

        if config is None:
            if len(self.label_list) > 1:
                config = {
                    "C": [1, 2, 5, 10, 20, 100],
                    "kernel": ["linear", "rbf", "poly"],
                    "max_cross_validation_folds": 5,
                    "seed": None,
                }
            else:
                config = {
                    "nu": 0.1,
                    "kernel": "rbf",
                }

        self.config = config

    def set_classification_model(self, config: dict = None):
        """Set and fit the SVC model.

        Args:
            config (dict, optional): A config containing keys for SVC kernels, C, max_cross_validation_folds.
                Defaults to None if self.config needs to be used.
        """
        if config:  # update if overwritten
            self.config = config

        if len(self.label_list) > 1:
            self.svm = SVC(
                probability=True,
                class_weight="balanced",
                verbose=self.verbose,
                random_state=self.config.get("seed"),
            )

            # NOTE: consifer using multi_target_strategy "one-vs-one", "one-vs-rest", "output-code"
            if self.multi_label:
                self.svm = OneVsRestClassifier(self.svm)
                param_addition = "estimator__"
                cv_splits = None
            else:
                param_addition = ""
                folds = self.config["max_cross_validation_folds"]
                cv_splits = max(2, min(folds, np.min(np.bincount(self.y)) // 5))

            tuned_parameters = [
                {
                    f"{param_addition}{key}": value
                    for key, value in self.config.items()
                    if key not in ["random_state", "max_cross_validation_folds", "seed"]
                }
            ]

            self.clf = GridSearchCV(
                self.svm,
                param_grid=tuned_parameters,
                n_jobs=1,
                cv=cv_splits,
                scoring="f1_weighted",
                verbose=self.verbose,
            )
            self.clf.fit(self.X, self.y)
        elif len(self.label_list) == 1:
            if self.multi_label:
                raise ValueError("Cannot apply one class classification with multiple-labels.")
            self.clf = OneClassSVM(verbose=self.verbose, **self.config)
            self.clf.fit(self.X)
        else:
            raise ValueError("Provide input data with Dict[key, List].")

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

        labels = []
        X = []
        self.label_list = list(self.data.keys())
        for key, value in self.data.items():
            labels += len(value) * [key]
            X += value

        if self.multi_label:
            df = pd.DataFrame(data={"X": X, "labels": labels})
            groups = df.groupby("X").agg(list).to_records().tolist()
            X = [group[0] for group in groups]
            labels = [group[1] for group in groups]
            self.le = preprocessing.MultiLabelBinarizer()
        else:
            self.le = preprocessing.LabelEncoder()

        self.y = self.le.fit_transform(labels)

        self.X = self.get_embeddings(X)

        if data:  # update if overwritten
            self.set_classification_model()


class ClassyExternal:
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

        return self.encoder.encode(docs, show_progress_bar=self.verbose)

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
