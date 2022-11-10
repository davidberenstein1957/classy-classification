from math import isclose

import pytest

from classy_classification import classyClassifier
from classy_classification.examples.data import (
    training_data_single_class,
    validation_data,
)


@pytest.fixture
def standalone_single_class():
    classifier = classyClassifier(data=training_data_single_class, multi_label=True)
    return classifier


def test_standalone_single_class(standalone_single_class):
    pred = standalone_single_class(validation_data[0])
    assert isclose(sum(pred.values()), 1)

    preds = standalone_single_class.pipe(validation_data)
    for pred in preds:
        assert isclose(sum(pred.values()), 1)
