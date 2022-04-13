from classy_classification import classyClassifier

from .data import training_data, validation_data

classifier = classyClassifier(data=training_data)
print(classifier(validation_data[0]))
print(classifier.pipe(validation_data))
