import spacy

import classy_classification
from .data import training_data, validation_data

nlp = spacy.load('en_core_web_md') 
nlp.add_pipe('text_categorizer', config={'data': training_data, 'model': 'spacy'})
print(nlp(validation_data[0])._.cats)
print([doc._.cats for doc in nlp.pipe(validation_data)])
