import re
import pandas as pd


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_review_column(df):
    common_names = [
        'review', 'Review', 'REVIEW',
        'text', 'Text', 'TEXT',
        'comment', 'Comment', 'COMMENT',
        'description', 'Description',
        'feedback', 'Feedback',
        'content', 'Content',
        'body', 'Body',
    ]
    for name in common_names:
        if name in df.columns:
            return name

    text_cols = df.select_dtypes(include='object').columns
    if len(text_cols) > 0:
        avg_lengths = {col: df[col].astype(str).str.len().mean()
                       for col in text_cols}
        return max(avg_lengths, key=avg_lengths.get)

    return df.columns[0]


def load_file(uploaded_file):
    filename = uploaded_file.name

    if filename.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")

    review_col = detect_review_column(df)
    df['cleaned_text'] = df[review_col].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 10].reset_index(drop=True)

    return df, review_col