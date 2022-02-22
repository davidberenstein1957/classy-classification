import os

from spacy import util
from spacy.language import Language
from spacy.tokens import Doc

from .classifiers.sentence_transformer import \
    classySentenceTransformer as classyClassifier
from .classifiers.spacy_external import classySpacyExternal
from .classifiers.spacy_internal import classySpacyInternal

__all__ = [
    'classyClassifier',
    'classySpacyExternal',
    'classySpacyInternal'
]

@Language.factory(
    "text_categorizer",
    default_config={
        "data": None,
        "model": 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        "device": "cpu",
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
    device: str,
    config: dict,
):  
    if model == 'spacy':
        return classySpacyInternal(
            nlp=nlp,
            name=name,
            data=data,
            config=config
        )
    else:
        return classySpacyExternal(
            name=name,
            data=data,
            device=device,
            model=model,
            config=config
        )
    

