import streamlit as st
import langdetect
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -------------------------------------------
# Load Sentiment Model (RoBERTa - Twitter Sentiment)
# -------------------------------------------
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

translator = Translator()

# -------------------------------------------
# Predict Sentiment Function
# -------------------------------------------
def predict_sentiment(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    logits = model(**tokens).logits
    sentiment_idx = torch.argmax(logits).item()
    return ["Negative", "Neutral", "Positive"][sentiment_idx]


# -------------------------------------------
# Streamlit UI
# -------------------------------------------
st.set_page_config(page_title="Multilingual Sentiment Analyzer", layout="centered")
st.title("🌎 Multilingual Sentiment Analyzer")

st.write("Enter text in **any language** and I will detect language ➝ translate ➝ analyze sentiment.")

user_input = st.text_area("✏️ Enter text here:", placeholder="Type something... (English / Hindi / Gujarati / Spanish / etc.)")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Detect language
        detected_lang = langdetect.detect(user_input)

        # Translate to English (only if not English)
        translated_text = translator.translate(user_input, dest="en").text if detected_lang != "en" else user_input

        # Sentiment
        sentiment = predict_sentiment(translated_text)

        st.success(f"✅ Sentiment: **{sentiment}**")

        with st.expander("🔎 Details"):
            st.write(f"**Detected Language:** `{detected_lang}`")
            st.write(f"**Translated Text:** {translated_text}")
