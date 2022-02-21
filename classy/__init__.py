import os

from spacy import util
from spacy.language import Language
from spacy.tokens import Doc

from .classy_sentence_transformer import classySentenceTransformer as Classy
from .classy_spacy_internal import classySpacyInternal

__all__ = [
    'classySentence'
]

@Language.factory(
    "text_categorizer",
    default_config={
        "data": None,
        "model": 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        "config": {                      
            "C": [1, 2, 5, 10, 20, 100],
            "kernels": ["linear"],                              
            "max_cross_validation_folds": 5
        },
    },
)
def make_text_categorizer(
    nlp: Language,
    name: str,
    data: dict,
    model: str,
    config: dict,
):  
    return classySpacyInternal(
        name=name,
        data=data,
        model=model,
        config=config
    )

