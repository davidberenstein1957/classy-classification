import spacy

import classy_classification
from examples import training_data, validation_data

nlp = spacy.blank('en')
nlp.add_pipe('text_categorizer', config={'data': training_data})
print(nlp(validation_data[0])._.cats)
print([doc._.cats for doc in nlp.pipe(validation_data)])
