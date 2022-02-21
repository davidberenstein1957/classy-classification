from typing import List

from sentence_transformers import SentenceTransformer

from .classy_skeleton import classySkeleton


class classySentenceTransformer(classySkeleton):
    def __init__(self, 
            data: dict, 
            model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 
            config: dict={                              
                "C": [1, 2, 5, 10, 20, 100],
                "kernels": ["linear"],                              
                "max_cross_validation_folds": 5
            },
            name: str = 'text_categorizer'
        ):
        super().__init__(data=data, config=config)
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
    
    
    
    
    
        
    
    
