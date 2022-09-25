from typing import List, Union

import onnxruntime
from fast_sentence_transformers.txtai import HFOnnx
from onnxruntime import InferenceSession, SessionOptions
from transformers import AutoTokenizer

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

        onnx = HFOnnx()
        embeddings = onnx(self.model, "pooling", "embeddings.onnx", quantize=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        options = SessionOptions()
        onnxproviders = onnxruntime.get_available_providers()

        if self.device == "cpu":
            fast_onnxprovider = "CPUExecutionProvider"
        else:
            if "CUDAExecutionProvider" not in onnxproviders:
                print("Using CPU. Try installing 'onnxruntime-gpu' or 'fast-sentence-transformers[gpu]'.")
                fast_onnxprovider = "CPUExecutionProvider"
            else:
                fast_onnxprovider = "CUDAExecutionProvider"
        self.session = InferenceSession(embeddings, options, providers=[fast_onnxprovider])

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
        inputs = self.tokenizer(X, padding=True, truncation=True, max_length=512, return_tensors="pt")
        ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
        return self.session.run(None, ort_inputs)[0]
