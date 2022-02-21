import spacy

import classy_classification

example_1 = {
    'positive': [
        'This is amazing!',
        'What an awesome library.',
        'I really love classy classification.',
        'Wow, this way of classifying is so easy.',
        'We are so positive about this approach.'
    ],
    'negative': [
        'Boo, this is the worst package ever.',
        'This library truly sucks.',
        'Why is this library sooo bad?',
        'We used this package and we are so negative about it.',
        'What a weird and shitty approach!'
    ]
}
nlp = spacy.blank('en')
nlp.add_pipe('text_categorizer', config={'data': example_1})
print(nlp('This library truly sucks.')._.cats)
print(list(nlp.pipe(['This library truly sucks.']))[0]._.cats)
