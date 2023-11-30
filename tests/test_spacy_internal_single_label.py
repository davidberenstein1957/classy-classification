import pytest
import spacy

from classy_classification.examples.data import training_data_single_class


@pytest.fixture(params=["en_core_web_md", "en_core_web_trf"])
def spacy_internal_single_label(request):
    nlp = spacy.load(request.param)
    nlp.add_pipe("classy_classification", config={"data": training_data_single_class})
    return nlp


def test_spacy_internal_single_label(spacy_internal_single_label):
    _ = spacy_internal_single_label(training_data_single_class["politics"][0])
    _ = spacy_internal_single_label.pipe(training_data_single_class["politics"])
