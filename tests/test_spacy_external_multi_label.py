import pytest
import spacy

from classy_classification.examples.data import (
    training_data_multi_label,
    validation_data,
)


@pytest.fixture
def spacy_external_multi_label():
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "classy_classification",
        config={"data": training_data_multi_label, "include_sent": True, "multi_label": True},
    )
    return nlp


def test_spacy_external_multi_label(spacy_external_multi_label):
    doc = spacy_external_multi_label(validation_data[0])
    assert doc._.cats
    for sent in doc.sents:
        assert sent._.cats

    docs = spacy_external_multi_label.pipe(validation_data)
    for doc in docs:
        assert doc._.cats
        for sent in doc.sents:
            assert sent._.cats
