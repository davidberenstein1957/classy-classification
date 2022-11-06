import warnings
from typing import List, Union

import numpy as np
import onnxruntime
from fast_sentence_transformers.txtai import HFOnnx
from fast_sentence_transformers.txtai.text import Labels
from onnxruntime import InferenceSession, SessionOptions
from spacy import util
from spacy.tokens import Doc
from transformers import AutoTokenizer

from .classy_skeleton import classySkeletonFewShot, classySkeletonFewShotMultiLabel


class classySpacy(object):
    def sentence_pipe(self, doc: Doc):
        sent_docs = [sent.as_doc() for sent in doc.sents]
        inferred_sent_docs = self.pipe(iter(sent_docs), include_sent=False)
        for sent_doc, sent in zip(inferred_sent_docs, doc.sents):
            sent._.cats = sent_doc._.cats

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


class classySpacyExternal(classySpacy):
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

        return self.endcoder.encode(docs)

    def set_embedding_model(self, model: str = None, device: str = "cpu", onnx=False):
        """set the embedding model based on a sentencetransformer model or path

        Args:
            model (str, optional): the model name. Defaults to self.model, if no model is provided.
        """
        if model:  # update if overwritten
            self.model = model
        if device:
            self.device = device

        if onnx:
            from fast_sentence_transformers import FastSentenceTransformer

            if device == "gpu":
                self.encoder = FastSentenceTransformer(self.model, device=self.device, quantize=False)
            else:
                warnings.warn("Not using quantization because this is not enabled via ONNX.")
                self.encoder = FastSentenceTransformer(self.model, device=self.device, quantize=True)
        else:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.model, device=self.device)

        # onnx = HFOnnx()
        # embeddings = onnx(self.model, "pooling", "embeddings.onnx", quantize=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        # options = SessionOptions()
        # onnxproviders = onnxruntime.get_available_providers()

        # if self.device == "cpu":
        #     fast_onnxprovider = "CPUExecutionProvider"
        # else:
        #     if "CUDAExecutionProvider" not in onnxproviders:
        #         print("Using CPU. Try installing 'onnxruntime-gpu' or 'fast-sentence-transformers[gpu]'.")
        #         fast_onnxprovider = "CPUExecutionProvider"
        #     else:
        #         fast_onnxprovider = "CUDAExecutionProvider"
        # self.session = InferenceSession(embeddings, options, providers=[fast_onnxprovider])

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
