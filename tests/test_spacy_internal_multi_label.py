import pytest
import spacy

from classy_classification.examples.data import (
    training_data_multi_label,
    validation_data,
)


@pytest.fixture(params=["en_core_web_md", "en_core_web_trf"])
def spacy_internal_multi_label(request):
    nlp = spacy.load(request.param)
    nlp.add_pipe(
        "text_categorizer",
        config={"data": training_data_multi_label, "model": "spacy", "include_sent": True, "multi_label": True},
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
