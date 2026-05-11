import torch
import streamlit as st
from transformers import pipeline
from config import MODEL_NAME, BATCH_SIZE, MAX_LENGTH, LABEL_MAP


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        device=device,
        return_all_scores=False
    )
    return model


def predict_batch(texts, model):
    all_results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        raw_results = model(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        )
        for r in raw_results:
            mapped_label = LABEL_MAP.get(r['label'], r['label'])
            all_results.append({
                "sentiment":  mapped_label,
                "confidence": round(r['score'] * 100, 1)
            })
    return all_results


def run_sentiment_analysis(df, review_col, model):
    texts = df['cleaned_text'].tolist()
    results = predict_batch(texts, model)
    df['Sentiment']    = [r['sentiment']  for r in results]
    df['Confidence %'] = [r['confidence'] for r in results]
    return df