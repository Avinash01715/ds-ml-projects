import streamlit as st
import joblib
import nltk
import string
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Classifier")


# ---------------------------------------------------------------------------
# One-time NLTK downloads
# ---------------------------------------------------------------------------

@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)


download_nltk_data()


# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():

    # Get the folder where app.py is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Build correct paths
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.joblib")
    model_path = os.path.join(BASE_DIR, "model.joblib")

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    return vectorizer, model


vectorizer, model = load_artifacts()


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def transform_text(text: str) -> str:
    """Lowercase → tokenize → remove non-alnum → remove stopwords → stem."""

    text = text.lower()

    tokens = nltk.word_tokenize(text)

    tokens = [
        w for w in tokens
        if w.isalnum()
    ]

    tokens = [
        w for w in tokens
        if w not in stop_words
        and w not in string.punctuation
    ]

    tokens = [
        ps.stem(w)
        for w in tokens
    ]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.markdown(
    "Paste or type an SMS message below and the model will predict "
    "whether it's **Ham** (legitimate) or **Spam**."
)

user_input = st.text_area(
    "Enter SMS message:",
    height=150,
    placeholder="e.g. Congratulations! You've won a £1000 prize. Call now..."
)


if st.button("Classify", type="primary"):

    if not user_input.strip():

        st.warning("Please enter a message first.")

    else:

        # Preprocess
        cleaned = transform_text(user_input)

        # Vectorize
        features = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(features)[0]

        probability = model.predict_proba(features)[0]

        # Display result
        if prediction == 1:

            st.error(
                f"🚨 **SPAM** "
                f"(confidence: {probability[1]:.1%})"
            )

        else:

            st.success(
                f"✅ **HAM** — legitimate message "
                f"(confidence: {probability[0]:.1%})"
            )

        # Debug information
        with st.expander("Debug: see preprocessed text"):
            st.code(cleaned)