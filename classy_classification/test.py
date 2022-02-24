from transformers import pipeline

from .classifiers.spacy_zero_shot_external import classySpacyZeroShotExternal


def test_individual_transformer():
    from .examples import individual_transformer
    
def test_spacy_few_shot_external():
    from .examples import spacy_few_shot_external
    
def test_spacy_zero_shot_external():
    from .examples import spacy_zero_shot_external
    
def test_spacy_internal_embeddings():
    from .examples import spacy_internal_embeddings


