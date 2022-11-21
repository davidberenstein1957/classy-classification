import spacy

from classy_classification.examples.data import training_data

nlp = spacy.blank("en")
nlp.add_pipe(
    "text_categorizer",
    config={"data": list(training_data), "cat_type": "zero", "include_sent": True, "multi_label": True},
)
print(nlp("kitchen stuff")._.cats)
