from math import isclose

import pytest
import spacy

from classy_classification.examples.data import training_data, validation_data


@pytest.fixture
def spacy_external_zer_shot():
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "classy_classification", config={"data": list(training_data.keys()), "cat_type": "zero", "include_sent": True}
    )
    return nlp


def test_spacy_external_zero_shot(spacy_external_zer_shot):
    doc = spacy_external_zer_shot(validation_data[0])
    assert isclose(sum(doc._.cats.values()), 1, abs_tol=0.05)
    for sent in doc.sents:
        assert isclose(sum(sent._.cats.values()), 1, abs_tol=0.05)


def test_spacy_external_zero_shot_bulk(spacy_external_zer_shot):
    docs = spacy_external_zer_shot.pipe(validation_data)
    for doc in docs:
        assert isclose(sum(doc._.cats.values()), 1, abs_tol=0.05)
        for sent in doc.sents:
            assert isclose(sum(sent._.cats.values()), 1, abs_tol=0.05)
