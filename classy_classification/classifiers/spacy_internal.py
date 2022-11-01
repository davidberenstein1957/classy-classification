from typing import List

from spacy import util
from spacy.tokens import Doc

from .classy_skeleton import classySkeletonFewShot, classySkeletonFewShotMultiLabel


class classySpacyInternal(object):
    def get_embeddings(self, text: List[str]) -> List[float]:
        """Retrieve embeddings from text.
        Overwrites function from the classySkeleton that is used to get embeddings for training data.

        Args:
            text (List[str]): a list of texts

        Returns:
            List[float]: a list of embeddings
        """
        docs = self.nlp.pipe(text)
        embeddings = [self.get_embeddings_from_doc(doc) for doc in docs]

        return embeddings

    def get_embeddings_from_doc(self, doc: Doc) -> List[float]:
        """Retrieve a vector from a spacy doc and internal embeddings.

        Args:
            doc (Doc): a spacy doc

        Raises:
            NotImplementedError: if not embeddings are present i.e. a trf or sm spacy model is used.

        Returns:
            List[float]: a vector embedding
        """
        if doc.has_vector:
            return doc.vector
        else:
            raise NotImplementedError(
                "internal spacy embeddings need to be derived from md/lg spacy models not from sm/trf models."
            )

    def set_pred_results_for_doc(self, doc: Doc):
        """
        > For each sentence in the document, get the embedding, then get the prediction for that embedding, then set the
        prediction as a property of the sentence

        :param doc: Doc
        :type doc: Doc
        :return: The doc object with the predicted results.
        """
        embeddings = [sent.as_doc().vector for sent in list(doc.sents)]
        pred_results = self.get_prediction(embeddings)
        for sent, pred in zip(doc.sents, pred_results):
            sent._.cats = pred
        return doc

    def __call__(self, doc: Doc):
        """
        It takes a doc, gets the embeddings from the doc, reshapes the embeddings, gets the prediction from the embeddings,
        and then sets the prediction results for the doc

        :param doc: Doc
        :type doc: Doc
        :return: The doc object with the predicted categories and the predicted categories for each sentence.
        """
        if self.include_doc:
            embeddings = self.get_embeddings_from_doc(doc)
            embeddings = embeddings.reshape(1, -1)
            doc._.cats = self.get_prediction(embeddings)[0]
        if self.include_sent:
            doc = self.set_pred_results_for_doc(doc)

        return doc

    def pipe(self, stream, batch_size=128):
        """
        predict the class for a spacy Doc stream

        Args:
            stream (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        for docs in util.minibatch(stream, size=batch_size):
            pred_results = [self.get_embeddings_from_doc(doc) for doc in docs]
            if self.include_doc:
                pred_results = self.get_prediction(pred_results)

            for doc, pred_result in zip(docs, pred_results):
                if self.include_doc:
                    doc._.cats = pred_result
                if self.include_sent:
                    doc = self.set_pred_results_for_doc(doc)

                yield doc


class classySpacyInternalFewShot(classySpacyInternal, classySkeletonFewShot):
    def __init__(self, *args, **kwargs):
        classySkeletonFewShot.__init__(self, *args, **kwargs)


class classySpacyInternalFewShotMultiLabel(classySpacyInternal, classySkeletonFewShotMultiLabel):
    def __init__(self, *args, **kwargs):
        classySkeletonFewShotMultiLabel.__init__(self, *args, **kwargs)

    # def set_classification_model(self, config: dict = None):
    #     """Set and fit the Multi-layer Perceptron (MLP) classifier.

    #     Args:
    #         config (dict, optional): A config for MLPClassifier: hidden_layer_sizes, seed.
    #     """
    #     if config:  # update if overwritten
    #         self.config = config

    #     hidden_layer_sizes = self.config["hidden_layer_sizes"]
    #     seed = self.config["seed"]
    #     self.clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=seed)
    #     self.clf.fit(self.X, self.y)

    # def proba_to_dict(self, pred_results: List[List]) -> List[dict]:
    #     """
    #     > It takes a list of lists of probabilities and returns a list of dictionaries where each dictionary has the label as
    #     the key and the probability as the value

    #     :param pred_results: List[List]
    #     :type pred_results: List[List]
    #     :return: A list of dictionaries.
    #     """
    #     pred_dict = []
    #     for pred in pred_results:
    #         pred_dict.append({label: value for label, value in zip(self.data.keys(), pred)})

    #     return pred_dict

    # def set_training_data(self, data: dict = None):
    #     """
    #     The function takes in a dictionary of data, and sets the training data for the model

    #     :param data: a dictionary of lists of strings. The keys are the labels, and the values are the samples
    #     :type data: dict
    #     """
    #     if data:  # update if overwritten
    #         self.data = data

    #     if data:  # update if overwritten
    #         self.set_classification_model()

    #     X = np.unique([sample for values in self.data.values() for sample in values])
    #     self.X = self.get_embeddings(X.tolist())
    #     self.y = [[1 if sample in values else 0 for values in self.data.values()] for sample in X]
