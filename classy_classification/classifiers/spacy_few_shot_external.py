from spacy import util
from spacy.tokens import Doc, Span

from .sentence_transformer import classySentenceTransformer


class classySpacyFewShotExternal(classySentenceTransformer):
    def __init__(self, nlp, name, data, device, config, include_doc, include_sent, *args, **kwargs):
        super().__init__(data=data, device=device, config=config, *args, **kwargs)
        self.name = name
        self.include_doc = include_doc
        self.include_sent = include_sent
        if include_sent:
            Span.set_extension("cats", default=None, force=True)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        if include_doc:
            Doc.set_extension("cats", default=None, force=True)

    def __call__(self, doc: Doc) -> Doc:
        """
        predict the class for a spacy Doc

        Args:
            doc (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        if self.include_doc:
            pred_result = super(self.__class__, self).__call__(doc.text.replace("\n", " "))
            doc._.cats = pred_result
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
            pred_results = [doc.text.replace("\n", " ") for doc in docs]

            if self.include_doc:
                pred_results = super(self.__class__, self).pipe(pred_results)

            for doc, pred_result in zip(docs, pred_results):
                if self.include_doc:
                    doc._.cats = pred_result
                if self.include_sent:
                    doc = self.set_pred_results_for_doc(doc)
                yield doc

    def set_pred_results_for_doc(self, doc: Doc):
        """
        It takes a spaCy Doc object, runs the text of each sentence through the model, and then adds the results to the Doc
        object

        :param doc: Doc
        :type doc: Doc
        :return: A list of dictionaries.
        """
        pred_results = super(self.__class__, self).pipe([sent.text for sent in list(doc.sents)])
        for sent, pred in zip(doc.sents, pred_results):
            sent._.cats = pred

        return doc
