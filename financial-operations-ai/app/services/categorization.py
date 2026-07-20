from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


TRAINING_EXAMPLES = {
    "Software": [
        "cloud hosting subscription",
        "software licenses and seats",
        "developer platform monthly plan",
        "cybersecurity monitoring service",
        "database hosting renewal",
        "team collaboration software",
    ],
    "Travel": [
        "hotel accommodation for client visit",
        "train tickets for conference",
        "flight and airport transfer",
        "business travel lodging",
        "rental car for site inspection",
        "conference travel expenses",
    ],
    "Office": [
        "office chairs and desks",
        "printer supplies and paper",
        "workplace coffee and supplies",
        "computer peripherals for office",
        "stationery order",
        "meeting room equipment",
    ],
    "Logistics": [
        "freight delivery and handling",
        "warehouse storage fee",
        "courier and parcel shipment",
        "international shipping charge",
        "distribution centre services",
        "equipment transport",
    ],
    "Professional Services": [
        "legal advisory engagement",
        "accounting and audit support",
        "management consulting services",
        "technical specialist contract",
        "recruitment agency fee",
        "engineering consultancy invoice",
    ],
}


class InvoiceCategorizer:
    def __init__(self):
        texts = []
        labels = []
        for label, examples in TRAINING_EXAMPLES.items():
            texts.extend(examples)
            labels.extend([label] * len(examples))
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("model", LogisticRegression(max_iter=1000, random_state=17)),
            ]
        )
        self.pipeline.fit(texts, labels)

    def predict(self, text):
        return str(self.pipeline.predict([text])[0])

    def confidence(self, text):
        probabilities = self.pipeline.predict_proba([text])[0]
        return float(probabilities.max())


categorizer = InvoiceCategorizer()

