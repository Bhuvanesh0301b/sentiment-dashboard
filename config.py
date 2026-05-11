MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32
MAX_LENGTH = 512

LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "negative": "Negative",
    "neutral":  "Neutral",
    "positive": "Positive",
    "NEGATIVE": "Negative",
    "NEUTRAL":  "Neutral",
    "POSITIVE": "Positive",
}

SENTIMENT_COLORS = {
    "Positive": "#2ECC71",
    "Negative": "#E74C3C",
    "Neutral":  "#95A5A6",
}