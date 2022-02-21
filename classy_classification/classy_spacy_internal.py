import os

import numpy as np
from spacy import util
from spacy.language import Language
from spacy.tokens import Doc
from torch import embedding

from .classy_skeleton import classySkeleton


class classySpacyInternal(classySkeleton):
    def __init__(self, nlp, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Doc.set_extension("cats", default=None, force=True)
        self.name = name
        self.nlp = nlp
        self.set_training_data()
        self.set_svc()

    def get_embeddings(self, text):
        docs = self.nlp.pipe(text)
        embeddings = [self.get_embeddings_from_doc(doc) for doc in docs]
        
        return embeddings
    
    def get_embeddings_from_doc(self, doc):
        return doc.vector if doc.has_vector else doc._.trf_data.tensors[-1][0]

    def __call__(self, doc: Doc):
        embeddings = self.get_embeddings_from_doc(doc)
        embeddings = embeddings.reshape(1, -1)
        doc._.cats = self.get_prediction(embeddings)[0]

        return doc

    def pipe(self, stream, batch_size=128):
        for docs in util.minibatch(stream, size=batch_size):
            embeddings = [self.get_embeddings_from_doc(doc) for doc in docs]
            pred_results = self.get_prediction(embeddings)
            
            for doc, pred_result in zip(docs, pred_results):
                doc._.cats = pred_result
                
                yield doc
  