from math import isclose

import pytest
import spacy

from classy_classification.examples.data import training_data, validation_data


@pytest.fixture
def spacy_external():
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "classy_classification",
        config={
            "data": training_data,
            "include_sent": True,
        },
    )
    return nlp


def test_spacy_external(spacy_external):
    doc = spacy_external(validation_data[0])
    assert isclose(sum(doc._.cats.values()), 1)
    for sent in doc.sents:
        assert isclose(sum(sent._.cats.values()), 1)

    docs = spacy_external.pipe(validation_data)
    for doc in docs:
        assert isclose(sum(doc._.cats.values()), 1)
        for sent in doc.sents:
            assert isclose(sum(sent._.cats.values()), 1)
