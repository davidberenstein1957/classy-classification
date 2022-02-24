from typing import List

from spacy import util
from spacy.tokens import Doc

from .classy_skeleton import classySkeleton


class classySpacyInternal(classySkeleton):
    def __init__(self, nlp, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Doc.set_extension("cats", default=None, force=True)
        self.name = name
        self.nlp = nlp
        self.set_training_data()
        self.set_svc()

    def get_embeddings(self, text: List[str]) -> List[float]:
        """ Retrieve embeddings from text.
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
        predict the class for a spacy Doc

        Args:
            doc (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        embeddings = self.get_embeddings_from_doc(doc)
        embeddings = embeddings.reshape(1, -1)
        doc._.cats = self.get_prediction(embeddings)[0]

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
            embeddings = [self.get_embeddings_from_doc(doc) for doc in docs]
            pred_results = self.get_prediction(embeddings)
            
            for doc, pred_result in zip(docs, pred_results):
                doc._.cats = pred_result
                
                yield doc
  