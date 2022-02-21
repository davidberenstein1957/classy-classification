import spacy

import .classy

old_data = {
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
nlp.add_pipe('text_categorizer', config={'data': old_data})
nlp('This library truly sucks.')
nlp.pipe(['This library truly sucks.'])

