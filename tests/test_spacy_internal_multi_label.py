import pytest
import spacy

from classy_classification.examples.data import training_data, validation_data


@pytest.fixture
def spacy_internal_multi_label():
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe(
        "text_categorizer", config={"data": training_data, "model": "spacy", "include_sent": True, "multi_label": True}
    )
    return nlp


def test_spacy_internal_multi_label(spacy_internal_multi_label):
    doc = spacy_internal_multi_label(validation_data[0])
    assert doc._.cats
    for sent in doc.sents:
        assert sent._.cats

    docs = spacy_internal_multi_label.pipe(validation_data)
    for doc in docs:
        assert doc._.cats
        for sent in doc.sents:
            assert sent._.cats
