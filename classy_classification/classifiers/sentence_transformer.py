from typing import List, Union

from sentence_transformers import SentenceTransformer

from .classy_skeleton import classySkeleton


class classySentenceTransformer(classySkeleton):
    def __init__(
        self,
        data: dict,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        config: Union[dict, None] = None,
    ):
        super().__init__(data=data, config=config)
        self.model = model
        self.device = device
        self.set_embedding_model()
        self.set_training_data()
        self.set_svc()

    def set_embedding_model(self, model: str = None, device: str = "cpu"):
        """set the embedding model based on a sentencetransformer model or path

        Args:
            model (str, optional): the model name. Defaults to self.model, if no model is provided.
        """
        if model:  # update if overwritten
            self.model = model
        if device:
            self.device = device

        self.embedding_model = SentenceTransformer(self.model, device=self.device)

        if model:  # update if overwritten
            self.set_training_data()
            self.set_svc()

    def get_embeddings(self, X: List[str]) -> List[List[float]]:
        """retrieve embeddings from the SentenceTransformer model for a text or list of texts

        Args:
            X (List[str]): input texts

        Returns:
            List[List[float]]: output embeddings
        """
        return self.embedding_model.encode(X)
