import os

from spacy import util
from spacy.language import Language
from spacy.tokens import Doc

from .sentence_transformer import classySentenceTransformer


class classySpacyExternal(classySentenceTransformer):
    def __init__(self, name,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        Doc.set_extension("cats", default=None, force=True)

    def __call__(self, doc: Doc) -> Doc:
        """
        predict the class for a spacy Doc

        Args:
            doc (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        pred_result = super(self.__class__, self).__call__(doc.text.replace("\n", " "))
        doc._.cats = pred_result

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
            texts = [doc.text.replace("\n", " ") for doc in docs]
            pred_results = super(self.__class__, self).pipe(texts)
            
            for doc, pred_result in zip(docs, pred_results):
                doc._.cats = pred_result
                
                yield doc
  