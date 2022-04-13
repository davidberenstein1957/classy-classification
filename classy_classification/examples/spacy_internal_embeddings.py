import spacy

import classy_classification  # noqa: F401

from .data import training_data, validation_data

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("text_categorizer", config={"data": training_data, "model": "spacy", "include_sent": True})
print([sent._.cats for sent in nlp(validation_data[0]).sents])
print([doc._.cats for doc in nlp.pipe(validation_data)])
