import pytest
import spacy

from classy_classification.examples.data import training_data_single_class


@pytest.fixture
def spacy_external_single_label():
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "classy_classification",
        config={"data": training_data_single_class},
    )
    return nlp


def test_spacy_external_single_label(spacy_external_single_label):
    _ = spacy_external_single_label(training_data_single_class["politics"][0])
    _ = spacy_external_single_label.pipe(training_data_single_class["politics"])
