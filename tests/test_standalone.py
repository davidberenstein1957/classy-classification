from math import isclose

import pytest

from classy_classification import ClassyClassifier
from classy_classification.examples.data import training_data, validation_data


@pytest.fixture
def standalone():
    classifier = ClassyClassifier(data=training_data)
    return classifier


def test_standalone(standalone):
    pred = standalone(validation_data[0])
    assert isclose(sum(pred.values()), 1)

    preds = standalone.pipe(validation_data)
    for pred in preds:
        assert isclose(sum(pred.values()), 1)
