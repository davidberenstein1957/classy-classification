import importlib.util
import pathlib
from typing import List, Union

import numpy as np
from spacy import util
from spacy.tokens import Doc

from .classy_skeleton import (
    classyExternal,
    classySkeleton,
    classySkeletonFewShot,
    classySkeletonFewShotMultiLabel,
)


class classySpacy:
    def sentence_pipe(self, doc: Doc):
        sent_docs = [sent.as_doc() for sent in doc.sents]
        inferred_sent_docs = self.pipe(iter(sent_docs), include_sent=False)
        for sent_doc, sent in zip(inferred_sent_docs, doc.sents):
            sent._.cats = sent_doc._.cats

    def __call__(self, doc: Doc):
        """
        It takes a doc, gets the embeddings from the doc, reshapes the embeddings, gets the prediction from the embeddings,
        and then sets the prediction results for the doc

        :param doc: Doc
        :type doc: Doc
        :return: The doc object with the predicted categories and the predicted categories for each sentence.
        """
        if self.include_doc:
            embeddings = self.get_embeddings([doc])
            embeddings = embeddings.reshape(1, -1)
            doc._.cats = self.get_prediction(embeddings)[0]

        if self.include_sent:
            self.sentence_pipe(doc)

        return doc

    def pipe(self, stream, batch_size=128, include_sent=None):
        """
        predict the class for a spacy Doc stream

        Args:
            stream (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        if include_sent is None:
            include_sent = self.include_sent
        for docs in util.minibatch(stream, size=batch_size):
            embeddings = self.get_embeddings(docs)
            pred_results = [] * len(embeddings)
            if self.include_doc:
                pred_results = self.get_prediction(embeddings)

            for doc, pred_result in zip(docs, pred_results):
                if self.include_doc:
                    doc._.cats = pred_result
                if include_sent:
                    self.sentence_pipe(doc)

                yield doc


class classySpacyInternal(classySpacy):
    def get_embeddings(self, docs: Union[List[Doc], List[str]]) -> List[float]:
        """Retrieve embeddings from text.
        Overwrites function from the classySkeleton that is used to get embeddings for training data to fetch internal
        spaCy embeddings.

        Args:
            text (List[str]): a list of texts

        Returns:
            List[float]: a list of embeddings
        """
        if isinstance(docs, list):
            if isinstance(docs[0], str):
                docs = self.nlp.pipe(docs, disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
            elif isinstance(docs[0], Doc):
                pass
        else:
            raise ValueError("This should be a List")

        embeddings = []
        for doc in docs:
            if doc.has_vector:
                embeddings.append(doc.vector)
            else:
                raise NotImplementedError(
                    "internal spacy embeddings need to be derived from md/lg spacy models not from sm/trf models."
                )

        return np.array(embeddings)


class classySpacyInternalFewShot(classySpacyInternal, classySkeletonFewShot):
    def __init__(self, *args, **kwargs):
        classySkeletonFewShot.__init__(self, *args, **kwargs)


class classySpacyInternalFewShotMultiLabel(classySpacyInternal, classySkeletonFewShotMultiLabel):
    def __init__(self, *args, **kwargs):
        classySkeletonFewShotMultiLabel.__init__(self, *args, **kwargs)


class classySpacyExternalFewShot(classySpacy, classyExternal, classySkeletonFewShot):
    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.set_embedding_model()
        classySkeletonFewShot.__init__(self, *args, **kwargs)


class classySpacyExternalFewShotMultiLabel(classySpacy, classyExternal, classySkeletonFewShotMultiLabel):
    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.set_embedding_model()
        classySkeletonFewShotMultiLabel.__init__(self, *args, **kwargs)


class classySpacyExternalZeroShot(classySpacy, classySkeleton):
    def __init__(
        self,
        model: str = "typeform/distilbert-base-uncased-mnli",
        device: str = "cpu",
        multi_label: bool = False,
        *args,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.multi_label = multi_label
        classySkeleton.__init__(self, *args, **kwargs)

    def set_classification_model(self, model: str = None, device: str = None):
        """set the embedding model based on a sentencetransformer model or path

        Args:
            model (str, optional): the model name. Defaults to self.model, if no model is provided.
        """
        if model:  # update if overwritten
            self.model = model
        if device:
            self.device = device

        if importlib.util.find_spec("fast-sentence-transformers") is None:
            from transformers import pipeline

            if self.device in ["gpu", "cuda", 0]:
                self.device = 0
            else:
                self.device = -1
            self.pipeline = pipeline("zero-shot-classification", model=self.model, device=self.device, top_k=None)
        else:
            from fast_sentence_transformers.txtai import HFOnnx
            from fast_sentence_transformers.txtai.text import Labels

            # Export model to ONNX
            p = pathlib.Path(self.model)
            if p.parent:
                p.parent.mkdir(parents=True, exist_ok=True)
            onnx = HFOnnx()
            onnx_model = onnx(self.model, "text-classification", f"{self.model}.onnx", quantize=False)

            # Run inference and validate
            if self.device in ["gpu", "cuda", 0]:
                self.pipeline = Labels((onnx_model, self.model), dynamic=True, gpu=True)
            else:
                self.pipeline = Labels((onnx_model, self.model), dynamic=True)

    def set_config(self, _: dict = None):
        """Zero-shot models don't require a config"""
        pass

    def set_training_data(self, _: dict = None):
        """Zero-shot models don't require training data"""
        pass

    def set_embedding_model(self, _: dict = None):
        """Zero-shot models don't require embeddings models"""
        pass

    def get_embeddings(self, docs: Union[List[Doc], List[str]]):
        """Zero-shot models don't require embeddings"""
        pass

    def format_prediction(self, prediction):
        """
        It takes a prediction dictionary and returns a list of dictionaries, where each dictionary has a single key-value
        pair

        :param prediction: The prediction returned by the model
        :return: A list of dictionaries.
        """
        if importlib.util.find_spec("fast-sentence-transformers") is None:
            return {pred[0]: pred[1] for pred in zip(prediction.get("labels"), prediction.get("scores"))}
        else:
            return {self.data[pred[0]]: pred[1] for pred in prediction}

    def set_pred_results_for_doc(self, doc: Doc):
        """
        It takes a spaCy Doc object, runs it through the pipeline, and then adds the predictions to the Doc object

        :param doc: Doc
        :type doc: Doc
        :return: A list of dictionaries.
        """
        pred_results = self.pipeline([sent.text for sent in list(doc.sents)], self.data)
        pred_results = [self.format_prediction(pred) for pred in pred_results]
        for sent, pred in zip(doc.sents, pred_results):
            sent._.cats = pred
        return doc

    def __call__(self, doc: Doc) -> Doc:
        """
        predict the class for a spacy Doc

        Args:
            doc (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        if self.include_doc:
            pred_result = self.pipeline(doc.text, self.data, multi_label=self.multi_label)
            doc._.cats = self.format_prediction(pred_result)
        if self.include_sent:
            self.sentence_pipe(doc)

        return doc

    def pipe(self, stream, batch_size=128, include_sent=None):
        """
        predict the class for a spacy Doc stream

        Args:
            stream (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        if include_sent is None:
            include_sent = self.include_sent
        for docs in util.minibatch(stream, size=batch_size):
            predictions = [doc.text for doc in docs]
            if self.include_doc:
                predictions = self.pipeline(predictions, self.data, multi_label=self.multi_label)
                predictions = [self.format_prediction(pred) for pred in predictions]
            for doc, pred_result in zip(docs, predictions):
                if self.include_doc:
                    doc._.cats = pred_result
                if include_sent:
                    self.sentence_pipe(doc)

                yield doc
