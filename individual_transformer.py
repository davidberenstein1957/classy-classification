from classy_classification import classyClassifier

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

classifier = classyClassifier(data=example_1)
print(classifier('This package is amazing!'))
print(classifier.pipe(['This package is the worst!']))

