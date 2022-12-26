import argilla as rg
from argilla._constants import DEFAULT_API_KEY
from transformers import pipeline

rg.init(api_url="http://localhost:6900", api_key=DEFAULT_API_KEY)

# load original dataset
dataset = rg.load(name="trec_with_active_learning")
print(len(dataset))
# load model and ensure use `top_k` to ensureall labels are returned during prediction
classifier = pipeline(
    "zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", framework="pt", top_k=None
)

# retrieve texts from record and batch-process prediction
labels = ["police", "officer"]
texts = [rec.text for rec in dataset]
predictions = classifier(texts, candidate_labels=labels)
print(len(predictions))
# check if prediction and data are correct
assert len(predictions) == len(dataset)

# overwrite records from original dataset to keep same id in rec.id
for pred, rec in zip(predictions, dataset):
    formatted_pred = list(zip(pred["labels"], pred["scores"]))
    rec.prediction = formatted_pred
    rec.prediction_agent = "https://huggingface.co/typeform/squeezebert-mnli"
    rec.metadata = {"split": "train"}

# upload updated records
rg.log(rec, name="policeofficer")
