import pytest

from classy_classification import ClassyClassifier
from classy_classification.examples.data import (
    training_data_multi_label,
    validation_data,
)


@pytest.fixture
def standalone_multi_label():
    classifier = ClassyClassifier(data=training_data_multi_label, multi_label=True)
    return classifier


def test_standalone_multi_label(standalone_multi_label):
    pred = standalone_multi_label(validation_data[0])
    assert pred

    preds = standalone_multi_label.pipe(validation_data)
    for pred in preds:
        assert pred
