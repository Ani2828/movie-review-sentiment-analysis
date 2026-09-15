import streamlit as st
import joblib
import re

# Load trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):
    # Remove HTML tags
    text = re.sub(r"<br\s*/?>", " ", text)

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review below and the machine learning model "
    "will predict whether the sentiment is positive or negative."
)

# User input
review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was absolutely fantastic..."
)

# Prediction
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:
        # Clean text
        cleaned_review = clean_text(review)

        # Convert text to TF-IDF
        review_tfidf = tfidf.transform([cleaned_review])

        # Prediction
        prediction = model.predict(review_tfidf)[0]

        # Probability
        probabilities = model.predict_proba(review_tfidf)[0]

        negative_probability = probabilities[0] * 100
        positive_probability = probabilities[1] * 100

        # Display result
        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.subheader("Prediction Confidence")

        st.write(
            f"Negative: **{negative_probability:.2f}%**"
        )

        st.write(
            f"Positive: **{positive_probability:.2f}%**"
        )

        st.progress(float(max(probabilities)))
