import classy_classification
import spacy

from .data import training_data, validation_data

nlp = spacy.blank('en')
nlp.add_pipe('text_categorizer', config={'data': list(training_data.keys()), 'cat_type': 'zero'})
print(nlp(validation_data[0])._.cats)
print([doc._.cats for doc in nlp.pipe(validation_data)])
