# Classy few shot classification
This repository contains an easy and intuitive approach to few-shot text classification. 

# Why?
[Huggingface](https://huggingface.co/) does offer some nice models for few/zero-shot classification, but these are not tailored to multi-lingual approaches. Rasa NLU has [a nice approach](https://rasa.com/blog/rasa-nlu-in-depth-part-1-intent-classification/) for this, but its to embedded in their codebase for easy usage outside of Rasa/chatbots. Additionally, it made sense to integrate [sentence-transformers](https://github.com/UKPLab/sentence-transformers), instead of default [word embeddings](https://arxiv.org/abs/1301.3781). Finally, I decided to integrate with Spacy, since training a custom [Spacy TextCategorizer](https://spacy.io/api/textcategorizer) seems like a lot of hassle if you want something quick. 

# Install

# Quickstart

# Inspiration Drawn From
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [Rasa NLU](https://github.com/RasaHQ/rasa) 
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Spacy](https://github.com/explosion/spaCy)
