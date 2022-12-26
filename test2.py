from sklearn.datasets import load_iris

from classy_classification import classyClassifier

X, y = load_iris(return_X_y=True)
print(type(X[0]))

data = {
    "furniture": [
        "This text is about chairs.",
        "Couches, benches and televisions.",
        "I really need to get a new sofa.",
        "There also exist things like fridges.",
        "I hope to be getting a new stove today.",
        "Do you also have some ovens.",
    ],
    "kitchen": [
        "There also exist things like fridges.",
        "I hope to be getting a new stove today.",
        "Do you also have some ovens.",
        "This text is about chairs.",
        "Couches, benches and televisions.",
        "I really need to get a new sofa.",
    ],
    "sport": ["we also cover sport a lot", "sports are amazing for the soccer"],
}

classifier = classyClassifier(data=data)
print(classifier("I am looking for furniture and kitchen equipment"))
