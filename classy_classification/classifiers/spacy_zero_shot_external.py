from fast_sentence_transformers.txtai import HFOnnx
from fast_sentence_transformers.txtai.text import Labels
from spacy import Language, util
from spacy.tokens import Doc, Span


class classySpacyZeroShotExternal(object):
    def __init__(
        self,
        nlp: Language,
        name: str,
        data: dict,
        model: str = "facebook/bart-large-mnli",
        device: str = "cpu",
        include_doc: bool = False,
        include_sent: bool = False,
    ):
        self.data = data
        self.name = name
        self.device = device
        self.model = model
        self.include_doc = include_doc
        self.include_sent = include_sent
        if include_sent:
            Span.set_extension("cats", default=None, force=True)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        if include_doc:
            Doc.set_extension("cats", default=None, force=True)
        self.set_classification_model()

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

    def pipe(self, stream, batch_size=128):
        """
        predict the class for a spacy Doc stream

        Args:
            stream (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        for docs in util.minibatch(stream, size=batch_size):
            predictions = [doc.text.replace("\n", " ") for doc in docs]
            if self.include_doc:
                predictions = self.pipeline(predictions, self.data)
                predictions = [self.format_prediction(pred) for pred in predictions]
            for doc, pred_result in zip(docs, predictions):
                if self.include_doc:
                    doc._.cats = pred_result
                if self.include_sent:
                    doc = self.set_pred_results_for_doc(doc)

                yield doc

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

    def format_prediction(self, prediction):
        """
        It takes a prediction dictionary and returns a list of dictionaries, where each dictionary has a single key-value
        pair

        :param prediction: The prediction returned by the model
        :return: A list of dictionaries.
        """
        return [{self.data[pred[0]]: pred[1] for pred in prediction}]
