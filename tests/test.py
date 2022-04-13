from transformers import pipeline

from ..classy_classification.classifiers.spacy_zero_shot_external import classySpacyZeroShotExternal


def test_individual_transformer():
    from ..classy_classification.examples import individual_transformer


def test_spacy_few_shot_external():
    from ..classy_classification.examples import spacy_few_shot_external


def test_spacy_zero_shot_external():
    from ..classy_classification.examples import spacy_zero_shot_external


def test_spacy_internal_embeddings():
    from ..classy_classification.examples import spacy_internal_embeddings
