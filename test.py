from classy_classification import classyClassifier

data = {
    "furniture": [
        "This text is about chairs.",
        "Couches, benches and televisions.",
        "I really need to get a new sofa.",
    ]
}
data_single = {
    "kitchen": [
        "There also exist things like fridges.",
        "I hope to be getting a new stove today.",
        "Do you also have some ovens.",
    ]
}

classifier = classyClassifier(data=data_single, multi_label=True)
print(classifier("Coke is a hell of a drug."))
classifier.pipe(["I am looking for kitchen appliances."])
