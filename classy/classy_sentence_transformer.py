from typing import List

from sentence_transformers import SentenceTransformer

from .classy_skeleton import classySkeleton


class classySentenceTransformer(classySkeleton):
    def __init__(self, model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.set_embedding_model()
        self.set_training_data()
        self.set_svc()
        
    def set_embedding_model(self, model: str = None):
        if model: # update if overwritten
            self.model = model
            
        self.embedding_model = SentenceTransformer(self.model)
        
        if model: # update if overwritten
            self.set_training_data()
            self.set_svc()  
            
    def get_embeddings(self, X):
        return self.embedding_model.encode(X)
    
    def __call__(self, text: str):
        embeddings = self.embedding_model.encode(text)
        embeddings = embeddings.reshape(1, -1)
        pred_result = self.clf.predict_proba(embeddings)
        
        return self.proba_to_dict(pred_result)
        
    def pipe(self, text: List[str]):
        embeddings = self.embedding_model.encode(text)
        pred_result = self.clf.predict_proba(embeddings)
        
        return self.proba_to_dict(pred_result)
    
    def proba_to_dict(self, pred_results):
        pred_dict = []
        for pred in pred_results:
            pred_dict.append(
                {label: value for label, value in zip(self.le.classes_, pred)}
            )
            
        return pred_dict
    
    
        
    
    
