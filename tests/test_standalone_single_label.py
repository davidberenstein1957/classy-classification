import pytest

from classy_classification import ClassyClassifier
from classy_classification.examples.data import training_data_single_class


@pytest.fixture
def standalone_single_label():
    classifier = ClassyClassifier(data=training_data_single_class)
    return classifier


def test_standalone_single_label(standalone_single_label):
    _ = standalone_single_label(training_data_single_class["politics"][0])
    _ = standalone_single_label.pipe(training_data_single_class["politics"])
