import pytest
import spacy

from classy_classification.examples.data import training_data, validation_data


@pytest.fixture
def nlp_multi_label():
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe(
        "text_categorizer",
        config={
            "data": training_data,
            "model": "spacy",
            "cat_type": "multi-label",
            "config": {"hidden_layer_sizes": (64,), "seed": 42},
        },
    )
    return nlp


def test_multi_label_prediction(nlp_multi_label):
    predictions = [doc._.cats for doc in nlp_multi_label.pipe(validation_data)]

    assert sum(predictions[0].values()) > 1
    assert predictions[0]["politics"] == 0.9385179048494932
    assert predictions[1]["sports"] == 0.47310880071826167
    assert predictions[2]["weather"] == 0.9921742105508168
