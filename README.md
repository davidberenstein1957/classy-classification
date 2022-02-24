# Classy few shot classification
This repository contains an easy and intuitive approach to zero and few-shot text classification using sentence-transformers, huggingface or word embeddings within a Spacy pipeline.
 
# Why?
[Huggingface](https://huggingface.co/) does offer some nice models for few/zero-shot classification, but these are not tailored to multi-lingual approaches. Rasa NLU has [a nice approach](https://rasa.com/blog/rasa-nlu-in-depth-part-1-intent-classification/) for this, but its too embedded in their codebase for easy usage outside of Rasa/chatbots. Additionally, it made sense to integrate [sentence-transformers](https://github.com/UKPLab/sentence-transformers) and [Hugginface zero-shot](https://huggingface.co/models?pipeline_tag=zero-shot-classification), instead of default [word embeddings](https://arxiv.org/abs/1301.3781). Finally, I decided to integrate with Spacy, since training a custom [Spacy TextCategorizer](https://spacy.io/api/textcategorizer) seems like a lot of hassle if you want something quick and dirty. 

# Install
``` pip install classy-classification```

# Quickstart
Take a look at the examples directory. Use data from any language. And choose a model from  [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) or from [Hugginface zero-shot](https://huggingface.co/models?pipeline_tag=zero-shot-classification).

```
from classy_classification import classyClassifier

data = {
    "furniture": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa."],
    "kitchen": ["There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens."]
}


classifier = classyClassifier(data)
classifier("I am looking for kitchen appliances.")
classifier.pipe(["I am looking for kitchen appliances."])

import spacy

nlp = spacy.blank("en")

nlp.add_pipe("text_categorizer", config={"data": data, "model": "spacy"}) # using spacy internal embeddings
nlp.add_pipe("text_categorizer", config={"data": data}) # using sentence-transformers
nlp.add_pipe("text_categorizer", config={"data": list(data.keys()), "cat_type": "zero"}) # using huggingface zero-shot


nlp("I am looking for kitchen appliances.")._.cats
nlp.pipe(["I am looking for kitchen appliances."])
```

# Credits
## Inspiration Drawn From
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [Rasa NLU](https://github.com/RasaHQ/rasa) 
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Spacy](https://github.com/explosion/spaCy)

## Or buy me a coffee
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/98kf2552674)


# More examples
## Some quick and dirty training data.
``` 
training_data = {
    "politics": [
        "Putin orders troops into pro-Russian regions of eastern Ukraine.",
        "The president decided not to go through with his speech.",
        "There is much uncertainty surrounding the coming elections.",
        "Democrats are engaged in a ‘new politics of evasion’."
    ],
    "sports": [
        "The soccer team lost.",
        "The team won by two against zero.",
        "I love all sport.",
        "The olympics were amazing.",
        "Yesterday, the tennis players wrapped up wimbledon."
    ],
    "weather": [
        "It is going to be sunny outside.",
        "Heavy rainfall and wind during the afternoon.",
        "Clear skies in the morning, but mist in the evenening.",
        "It is cold during the winter.",
        "There is going to be a storm with heavy rainfall."
    ]
}

validation_data = [
    "I am surely talking about politics.",
    "Sports is all you need.",
    "Weather is amazing."
]
```


## using an individual sentence-transformer
```
from classy_classification import classyClassifier

classifier = classyClassifier(data=training_data)
classifier(validation_data[0])
classifier.pipe(validation_data)

# overwrite training data
classifier.set_training_data(data=new_training_data)

# overwrite [embedding model](https://www.sbert.net/docs/pretrained_models.html)
classifier.set_embedding_model(model="paraphrase-MiniLM-L3-v2")

# overwrite SVC config
classifier.set_svc(
    config={                              
        "C": [1, 2, 5, 10, 20, 100],
        "kernels": ["linear"],                              
        "max_cross_validation_folds": 5
    }
)
```

## external sentence-transformer within spacy pipeline for few-shot
```
import spacy

import classy_classification

nlp = spacy.blank("en")
nlp.add_pipe("text_categorizer", config={"data": training_data}) #
nlp(validation_data[0])._.cats
nlp.pipe(validation_data)
```

## external hugginface model within spacy pipeline for zero-shot
```
import spacy

import classy_classification

nlp = spacy.blank("en")
nlp.add_pipe("text_categorizer", config={"data": training_data, "cat_type": "zero"}) #
nlp(validation_data[0])._.cats
nlp.pipe(validation_data)
```
## internal spacy word2vec embeddings
```
import spacy

import classy_classification

nlp = spacy.load("en_core_web_md") 
nlp.add_pipe("text_categorizer", config={"data": training_data, "model": "spacy"}) #use internal embeddings from spacy model
nlp(validation_data[0])._.cats
nlp.pipe(validation_data)
```

# Todo

[ ] look into a way to integrate spacy trf models.
[ ] multiple clasifications datasets for a single input e.g. emotions and topic.
