import classy_classification
import spacy

from .data import training_data, validation_data

# this has not been implemented yet


# nlp = spacy.load('en_core_web_trf')
# nlp.add_pipe('text_categorizer', config={'data': training_data, 'model': 'spacy'})
# print(nlp(validation_data[0])._.cats)
# print([doc._.cats for doc in nlp.pipe(validation_data)])
