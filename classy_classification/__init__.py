from typing import Union

from spacy.language import Language

from .classifiers.sentence_transformer import (
    classySentenceTransformer as classyClassifier,
)
from .classifiers.spacy_few_shot_external import classySpacyFewShotExternal
from .classifiers.spacy_internal import classySpacyInternal
from .classifiers.spacy_zero_shot_external import classySpacyZeroShotExternal

__all__ = ["classyClassifier", "classySpacyFewShotExternal", "classySpacyZeroShotExternal", "classySpacyInternal"]


@Language.factory(
    "text_categorizer",
    default_config={
        "data": None,
        "model": None,
        "device": "cpu",
        "config": {"C": [1, 2, 5, 10, 20, 100], "kernels": ["linear"], "max_cross_validation_folds": 5},
        "cat_type": "few",
        "include_doc": True,
        "include_sent": False,
    },
)
def make_text_categorizer(
    nlp: Language,
    name: str,
    data: Union[dict, list],
    device: str,
    config: dict,
    model: str = None,
    cat_type: str = "few",
    include_doc: bool = True,
    include_sent: bool = False,
):
    if model == "spacy":
        if cat_type == "zero":
            raise NotImplementedError("cannot use spacy internal embeddings with zero-shot classification")
        return classySpacyInternal(
            nlp=nlp, name=name, data=data, config=config, include_doc=include_doc, include_sent=include_sent
        )
    else:
        if cat_type == "zero":
            if model:
                return classySpacyZeroShotExternal(
                    nlp=nlp,
                    name=name,
                    data=data,
                    device=device,
                    model=model,
                    include_doc=include_doc,
                    include_sent=include_sent,
                )
            else:
                return classySpacyZeroShotExternal(
                    nlp=nlp,
                    name=name,
                    data=data,
                    device=device,
                    include_doc=include_doc,
                    include_sent=include_sent,
                )
        else:
            if model:
                return classySpacyFewShotExternal(
                    nlp=nlp,
                    name=name,
                    data=data,
                    device=device,
                    model=model,
                    config=config,
                    include_doc=include_doc,
                    include_sent=include_sent,
                )
            else:
                return classySpacyFewShotExternal(
                    nlp=nlp,
                    name=name,
                    data=data,
                    device=device,
                    config=config,
                    include_doc=include_doc,
                    include_sent=include_sent,
                )
