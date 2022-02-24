from spacy import util
from spacy.tokens import Doc
from transformers import pipeline


class classySpacyZeroShotExternal(object):
    def __init__(self, 
            name: str,
            data: dict, 
            model: str = 'facebook/bart-large-mnli', 
            device: str = 'cpu',
        ):
        self.data = data
        self.name = name
        self.device = device
        self.model = model
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
        pred_result = self.pipeline(doc.text, self.data)
        doc._.cats = self.format_prediction(pred_result)

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
            predictions = self.pipeline(texts, self.data)
            predictions = [self.format_prediction(pred) for pred in predictions]
            for doc, pred_result in zip(docs, predictions):
                doc._.cats = pred_result
                
                yield doc
        
    def set_classification_model(self, model: str = None, device: str = None):
        """ set the embedding model based on a sentencetransformer model or path

        Args:
            model (str, optional): the model name. Defaults to self.model, if no model is provided.
        """
        if model: # update if overwritten
            self.model = model
        if device:
            self.device = device
        
        if self.device == 'gpu':
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=self.model, 
                device=0
            )
        else:
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=self.model
            )
            
    @staticmethod
    def format_prediction(prediction):
        return [{label: score} for label, score in zip(prediction['labels'], prediction['scores'])]

        
    
    
    
    
    
    
        
    
    
