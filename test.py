import spacy

import classy_classification  # noqa: F401
from classy_classification.examples.data import training_data, validation_data

classifier = classy_classification.classyClassifier(data=training_data)
print(classifier(validation_data[0]))
print(classifier.pipe(validation_data))
