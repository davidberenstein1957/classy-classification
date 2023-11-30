from classy_classification import ClassyClassifier

from .data import training_data, validation_data

classifier = ClassyClassifier(data=training_data)
print(classifier(validation_data[0]))
print(classifier.pipe(validation_data))
