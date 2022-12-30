import logging
from typing import Union

from spacy.language import Language

from .classifiers.classy_spacy import (
    ClassySpacyExternalFewShot,
    ClassySpacyExternalZeroShot,
    ClassySpacyInternalFewShot,
)
from .classifiers.classy_standalone import ClassySentenceTransformer as ClassyClassifier

__all__ = [
    "ClassyClassifier",
    "ClassySpacyExternalFewShot",
    "ClassySpacyExternalZeroShot",
    "ClassySpacyInternalFewShot",
]

logging.captureWarnings(True)


@Language.factory(
    "text_categorizer",
    default_config={
        "data": None,
        "model": None,
        "device": "cpu",
        "config": None,
        "cat_type": "few",
        "multi_label": False,
        "include_doc": True,
        "include_sent": False,
        "verbose": False,
    },
)
def make_text_categorizer(
    nlp: Language,
    name: str,
    data: Union[dict, list],
    device: str,
    config: dict = None,
    model: str = None,
    cat_type: str = "few",
    multi_label: bool = False,
    include_doc: bool = True,
    include_sent: bool = False,
    verbose: bool = False,
):
    if model == "spacy" and cat_type == "zero":
        raise NotImplementedError("Cannot use spacy internal embeddings with zero-shot classification")
    elif model == "spacy" and cat_type == "few":
        return ClassySpacyInternalFewShot(
            nlp=nlp,
            name=name,
            data=data,
            config=config,
            include_doc=include_doc,
            include_sent=include_sent,
            multi_label=multi_label,
            verbose=verbose,
        )

    elif model != "spacy" and cat_type == "zero":
        return ClassySpacyExternalZeroShot(
            nlp=nlp,
            name=name,
            data=data,
            device=device,
            model=model,
            include_doc=include_doc,
            include_sent=include_sent,
            multi_label=multi_label,
            verbose=verbose,
        )
    elif model != "spacy" and cat_type == "few":
        return ClassySpacyExternalFewShot(
            nlp=nlp,
            name=name,
            data=data,
            device=device,
            model=model,
            config=config,
            include_doc=include_doc,
            include_sent=include_sent,
            multi_label=multi_label,
            verbose=verbose,
        )
    else:
        raise NotImplementedError(
            f"`model` as `{model}` is not valid it takes arguments `spacy` and `transformer`. "
            f"`cat_type` as `{cat_type}` is not valid stakes arguments `zero` and `few`."
        )
