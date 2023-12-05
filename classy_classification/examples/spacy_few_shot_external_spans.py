import spacy
from classy_classification.examples.data import training_data_spans, validation_data_spans
import classy_classification  # noqa: F401


nlp = spacy.load("en_core_web_md")

# Create a SpanRuler with a pattern for the word "weather" and its two preceding/succeeding tokens
span_ruler = nlp.add_pipe("span_ruler")
patterns = [{"label": "WEATHER", "pattern": [{}, {"LOWER": "weather"}, {}]}]
span_ruler.add_patterns(patterns)


nlp.add_pipe(
    "classy_classification",
    config={
        "data": training_data_spans,
        "include_doc": True,
        "include_spans": True,
        "model": "spacy",
    },
)

dc = nlp(validation_data_spans[0])
for doc in nlp.pipe(validation_data_spans):
    print(doc.spans["ruler"])
    print([span._.cats for span in doc.spans["ruler"]])
