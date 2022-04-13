from typing import List

from spacy import util
from spacy.tokens import Doc, Span

from .classy_skeleton import classySkeleton


class classySpacyInternal(classySkeleton):
    def __init__(self, nlp, name, data, config, include_doc, include_sent):
        super().__init__(data=data, config=config)
        self.include_doc = include_doc
        self.include_sent = include_sent
        if include_sent:
            Span.set_extension("cats", default=None, force=True)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        if include_doc:
            Doc.set_extension("cats", default=None, force=True)
        self.name = name
        self.nlp = nlp
        self.set_training_data()
        self.set_svc()

    def get_embeddings(self, text: List[str]) -> List[float]:
        """Retrieve embeddings from text.
        Overwrites function from the classySkeleton that is used to get embeddings for training data.

        Args:
            text (List[str]): a list of texts

        Returns:
            List[float]: a list of embeddings
        """
        docs = self.nlp.pipe(text)
        embeddings = [self.get_embeddings_from_doc(doc) for doc in docs]

        return embeddings

    def get_embeddings_from_doc(self, doc: Doc) -> List[float]:
        """Retrieve a vector from a spacy doc and internal embeddings.

        Args:
            doc (Doc): a spacy doc

        Raises:
            NotImplementedError: if not embeddings are present i.e. a trf or sm spacy model is used.

        Returns:
            List[float]: a vector embedding
        """
        if doc.has_vector:
            return doc.vector
        else:
            raise NotImplementedError(
                "internal spacy embeddings need to be derived from md/lg spacy models not from sm/trf models."
            )

    def __call__(self, doc: Doc):
        """
        It takes a doc, gets the embeddings from the doc, reshapes the embeddings, gets the prediction from the embeddings,
        and then sets the prediction results for the doc

        :param doc: Doc
        :type doc: Doc
        :return: The doc object with the predicted categories and the predicted categories for each sentence.
        """
        if self.include_doc:
            embeddings = self.get_embeddings_from_doc(doc)
            embeddings = embeddings.reshape(1, -1)
            doc._.cats = self.get_prediction(embeddings)[0]
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
            pred_results = [self.get_embeddings_from_doc(doc) for doc in docs]
            if self.include_doc:
                pred_results = self.get_prediction(pred_results)

            for doc, pred_result in zip(docs, pred_results):
                if self.include_doc:
                    doc._.cats = pred_result
                if self.include_sent:
                    doc = self.set_pred_results_for_doc(doc)

                yield doc

    def set_pred_results_for_doc(self, doc: Doc):
        """
        > For each sentence in the document, get the embedding, then get the prediction for that embedding, then set the
        prediction as a property of the sentence

        :param doc: Doc
        :type doc: Doc
        :return: The doc object with the predicted results.
        """
        embeddings = [sent.as_doc().vector for sent in list(doc.sents)]
        pred_results = self.get_prediction(embeddings)
        for sent, pred in zip(doc.sents, pred_results):
            sent._.cats = pred
        return doc
