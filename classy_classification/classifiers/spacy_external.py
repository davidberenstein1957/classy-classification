from typing import List

import onnxruntime
from fast_sentence_transformers.txtai import HFOnnx
from fast_sentence_transformers.txtai.text import Labels
from onnxruntime import InferenceSession, SessionOptions
from spacy import util
from spacy.tokens import Doc
from transformers import AutoTokenizer

from .classy_skeleton import (
    classySkeleton,
    classySkeletonFewShot,
    classySkeletonFewShotMultiLabel,
)


class classySpacyExternal(object):
    def get_embeddings(self, X: List[str]) -> List[List[float]]:
        """retrieve embeddings from the SentenceTransformer model for a text or list of texts

        Args:
            X (List[str]): input texts

        Returns:
            List[List[float]]: output embeddings
        """
        inputs = self.tokenizer(X, padding=True, truncation=True, max_length=512, return_tensors="pt")
        ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}

        return self.session.run(None, ort_inputs)[0]

    def set_embedding_model(self, model: str = None, device: str = "cpu"):
        """set the embedding model based on a sentencetransformer model or path

        Args:
            model (str, optional): the model name. Defaults to self.model, if no model is provided.
        """
        if model:  # update if overwritten
            self.model = model
        if device:
            self.device = device

        onnx = HFOnnx()
        embeddings = onnx(self.model, "pooling", "embeddings.onnx", quantize=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        options = SessionOptions()
        onnxproviders = onnxruntime.get_available_providers()

        if self.device == "cpu":
            fast_onnxprovider = "CPUExecutionProvider"
        else:
            if "CUDAExecutionProvider" not in onnxproviders:
                print("Using CPU. Try installing 'onnxruntime-gpu' or 'fast-sentence-transformers[gpu]'.")
                fast_onnxprovider = "CPUExecutionProvider"
            else:
                fast_onnxprovider = "CUDAExecutionProvider"
        self.session = InferenceSession(embeddings, options, providers=[fast_onnxprovider])

        if model:  # update if overwritten
            self.set_training_data()
            self.set_classification_model()


class classySpacyExternalFewShot(classySpacyExternal, classySkeletonFewShot):
    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        *args,
        **kwargs
    ):
        self.model = model
        self.device = device
        self.set_embedding_model()
        classySkeletonFewShot.__init__(self, *args, **kwargs)


class classySpacyExternalFewShotMultiLabel(classySpacyExternal, classySkeletonFewShotMultiLabel):
    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        *args,
        **kwargs
    ):
        self.model = model
        self.device = device
        self.set_embedding_model()
        classySkeletonFewShotMultiLabel.__init__(self, *args, **kwargs)


class classySpacyExternalZeroShot(classySkeleton):
    def __init__(self, model: str = "facebook/bart-large-mnli", device: str = "cpu", *args, **kwargs):
        self.model = model
        self.device = device
        super().__init__(*args, **kwargs)

    def set_classification_model(self, model: str = None, device: str = None):
        """set the embedding model based on a sentencetransformer model or path

        Args:
            model (str, optional): the model name. Defaults to self.model, if no model is provided.
        """
        if model:  # update if overwritten
            self.model = model
        if device:
            self.device = device

        # Export model to ONNX
        onnx = HFOnnx()
        onnx_model = onnx(self.model, "text-classification", "zero-shot.onnx", quantize=False)

        # Run inference and validate
        if self.device == "gpu":
            self.pipeline = Labels((onnx_model, self.model), dynamic=True, gpu=True)
        else:
            self.pipeline = Labels((onnx_model, self.model), dynamic=True)

    def set_training_data(self, data: dict = None):
        """Zero-shot doesn't require training data"""
        pass

    def format_prediction(self, prediction):
        """
        It takes a prediction dictionary and returns a list of dictionaries, where each dictionary has a single key-value
        pair

        :param prediction: The prediction returned by the model
        :return: A list of dictionaries.
        """
        return [{self.data[pred[0]]: pred[1] for pred in prediction}]

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
            pred_result = self.pipeline(doc.text, self.data)
            doc._.cats = self.format_prediction(pred_result)
        if self.include_sent:
            doc = self.set_pred_results_for_doc(doc)

        return doc

    def pipe(self, stream, batch_size=128):
        """
        predict the class for a spacy Doc stream

        Args:
            stream (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        for docs in util.minibatch(stream, size=batch_size):
            predictions = [doc.text for doc in docs]
            if self.include_doc:
                predictions = self.pipeline(predictions, self.data)
                predictions = [self.format_prediction(pred) for pred in predictions]
            for doc, pred_result in zip(docs, predictions):
                if self.include_doc:
                    doc._.cats = pred_result
                if self.include_sent:
                    doc = self.set_pred_results_for_doc(doc)

                yield doc
