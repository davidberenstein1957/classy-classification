# Classy Classification
Have you every struggled with needing a [Spacy TextCategorizer](https://spacy.io/api/textcategorizer) but didn't have the time to train one from scratch? Classy Classification is the way to go! For few-shot classification using [sentence-transformers](https://github.com/UKPLab/sentence-transformers) or [spaCy models](https://spacy.io/usage/models), provide a dictionary with labels and examples, or just provide a list of labels for zero shot-classification with [Hugginface zero-shot classifiers](https://huggingface.co/models?pipeline_tag=zero-shot-classification).

[![Current Release Version](https://img.shields.io/github/release/pandora-intelligence/classy-classification.svg?style=flat-square&logo=github)](https://github.com/pandora-intelligence/classy-classification/releases)
[![pypi Version](https://img.shields.io/pypi/v/classy-classification.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/classy-classification/)
[![PyPi downloads](https://static.pepy.tech/personalized-badge/classy-classification?period=total&units=international_system&left_color=grey&right_color=orange&left_text=pip%20downloads)](https://pypi.org/project/classy-classification/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

# Install
``` pip install classy-classification```

or install with faster inference using onnx.

``` pip install classy-classification[onnx]```

## ONNX issues

### pickling

ONNX does show some issues when pickling the data.
### M1
Some [installation issues](https://github.com/onnx/onnx/issues/3129) might occur, which can be fixed by these commands.

```
brew install cmake
brew install protobuf
pip3 install onnx --no-use-pep517
```

# Quickstart
## SpaCy embeddings
```python
import spacy
import classy_classification

data = {
    "furniture": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa."],
    "kitchen": ["There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens."]
}

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe(
    "text_categorizer",
    config={
        "data": data,
        "model": "spacy"
    }
)

print(nlp("I am looking for kitchen appliances.")._.cats)

# Output:
#
# [{"label": "furniture", "score": 0.21}, {"label": "kitchen", "score": 0.79}]
```
### Sentence level classification
```python
import spacy
import classy_classification

data = {
    "furniture": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa."],
    "kitchen": ["There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens."]
}

nlp.add_pipe(
    "text_categorizer",
    config={
        "data": data,
        "model": "spacy",
        "include_sent": True
    }
)

print(nlp("I am looking for kitchen appliances. And I love doing so.").sents[0]._.cats)

# Output:
#
# [{"label": "furniture", "score": 0.21}, {"label": "kitchen", "score": 0.79}]
```
### Define random seed and verbosity
```python

nlp.add_pipe(
    "text_categorizer",
    config={
        "data": data,
        "verbose": True,
        "config": {"seed": 42}
    }
)
```
### Multi-label classification
Sometimes multiple labels are necessary to fully describe the contents of a text. In that case, we want to make use of the **multi-label** implementation, here the sum of label scores is not limited to 1. Just pass the same training data to multiple keys.

```python
import spacy
import classy_classification

data = {
    "furniture": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa.",
               "We have a new dinner table.",
               "There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens.",
                "We have a new dinner table."],
    "kitchen": ["There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens.",
                "We have a new dinner table.",
                "There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens.",
                "We have a new dinner table."]
}

nlp = spacy.load("en_core_web_md")
nlp.add_pipe(
    "text_categorizer",
    config={
        "data": data,
        "model": "spacy",
        "multi_label": True,
    }
)

print(nlp("I am looking for furniture and kitchen equipment.")._.cats)

# Output:
#
# [{"label": "furniture", "score": 0.92}, {"label": "kitchen", "score": 0.91}]
```
### Outlier detection
Sometimes it is worth to be able to do outlier detection or binary classification. This can either be approached using
a binary training dataset, however, I have also implemented support for a `OneClassSVM` for [outlier detection using a single label](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html). Not that this method does not return probabilities, but that the data is formatted like label-score value pair to ensure uniformity.

Approach 1:
```python
import spacy
import classy_classification

data_binary = {
    "inlier": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa."],
    "outlier": ["Text about kitchen equipment",
                "This text is about politics",
                "Comments about AI and stuff."]
}

nlp = spacy.load("en_core_web_md")
nlp.add_pipe(
    "text_categorizer",
    config={
        "data": data_binary,
    }
)

print(nlp("This text is a random text")._.cats)

# Output:
#
# [{'inlier': 0.2926672385488411, 'outlier': 0.707332761451159}]
```
Approach 2:
```python
import spacy
import classy_classification

data_singular = {
    "furniture": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa.",
               "We have a new dinner table."]
}
nlp = spacy.load("en_core_web_md")
nlp.add_pipe(
    "text_categorizer",
    config={
        "data": data_singular,
    }
)

print(nlp("This text is a random text")._.cats)

# Output:
#
# [{'furniture': 0, 'not_furniture': 1}]
```
## Sentence-transfomer embeddings
```python
import spacy
import classy_classification

data = {
    "furniture": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa."],
    "kitchen": ["There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens."]
}

nlp = spacy.blank("en")
nlp.add_pipe(
    "text_categorizer",
    config={
        "data": data,
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "device": "gpu"
    }
)

print(nlp("I am looking for kitchen appliances.")._.cats)

# Output:
#
# [{"label": "furniture", "score": 0.21}, {"label": "kitchen", "score": 0.79}]
```
## Hugginface zero-shot classifiers
```python
import spacy
import classy_classification

data = ["furniture", "kitchen"]

nlp = spacy.blank("en")
nlp.add_pipe(
    "text_categorizer",
    config={
        "data": data,
        "model": "typeform/distilbert-base-uncased-mnli",
        "cat_type": "zero",
        "device": "gpu"
    }
)

print(nlp("I am looking for kitchen appliances.")._.cats)

# Output:
#
# [{"label": "furniture", "score": 0.21}, {"label": "kitchen", "score": 0.79}]
```
# Credits
## Inspiration Drawn From
[Huggingface](https://huggingface.co/) does offer some nice models for few/zero-shot classification, but these are not tailored to multi-lingual approaches. Rasa NLU has [a nice approach](https://rasa.com/blog/rasa-nlu-in-depth-part-1-intent-classification/) for this, but its too embedded in their codebase for easy usage outside of Rasa/chatbots. Additionally, it made sense to integrate [sentence-transformers](https://github.com/UKPLab/sentence-transformers) and [Hugginface zero-shot](https://huggingface.co/models?pipeline_tag=zero-shot-classification), instead of default [word embeddings](https://arxiv.org/abs/1301.3781). Finally, I decided to integrate with Spacy, since training a custom [Spacy TextCategorizer](https://spacy.io/api/textcategorizer) seems like a lot of hassle if you want something quick and dirty.

- [Scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [Rasa NLU](https://github.com/RasaHQ/rasa)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Spacy](https://github.com/explosion/spaCy)

## Or buy me a coffee
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/98kf2552674)


# Standalone usage without spaCy

```python

from classy_classification import classyClassifier

data = {
    "furniture": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa."],
    "kitchen": ["There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens."]
}

classifier = classyClassifier(data=data)
classifier("I am looking for kitchen appliances.")
classifier.pipe(["I am looking for kitchen appliances."])

# overwrite training data
classifier.set_training_data(data=data)
classifier("I am looking for kitchen appliances.")

# overwrite [embedding model](https://www.sbert.net/docs/pretrained_models.html)
classifier.set_embedding_model(model="paraphrase-MiniLM-L3-v2")
classifier("I am looking for kitchen appliances.")

# overwrite SVC config
classifier.set_classification_model(
    config={
        "C": [1, 2, 5, 10, 20, 100],
        "kernel": ["linear"],
        "max_cross_validation_folds": 5
    }
)
classifier("I am looking for kitchen appliances.")
```

## Save and load models
```python
data = {
    "furniture": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa."],
    "kitchen": ["There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens."]
}
classifier = classyClassifier(data=data)

with open("./classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

f = open("./classifier.pkl", "rb")
classifier = pickle.load(f)
classifier("I am looking for kitchen appliances.")
```
