import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

st.title("📩 SMS Spam Classifier")

# ---------------------------------------------------------------------------
# One-time NLTK downloads (cached by Streamlit so they only run once)
# ---------------------------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk_data()

# ---------------------------------------------------------------------------
# Load model artifacts (cached so they stay in memory across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("vectorizer.joblib")
    model = joblib.load("model.joblib")
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
    tokens = [w for w in tokens if w.isalnum()]
    tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    tokens = [ps.stem(w) for w in tokens]
    return " ".join(tokens)

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.markdown("Paste or type an SMS message below and the model will predict whether it's **Ham** (legitimate) or **Spam**.")

user_input = st.text_area("Enter SMS message:", height=150, placeholder="e.g. Congratulations! You've won a £1000 prize. Call now...")

if st.button("Classify", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message first.")
    else:
        # Preprocess → vectorize → predict
        cleaned = transform_text(user_input)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        if prediction == 1:
            st.error(f"🚨 **SPAM** (confidence: {probability[1]:.1%})")
        else:
            st.success(f"✅ **HAM** — legitimate message (confidence: {probability[0]:.1%})")

        with st.expander("Debug: see preprocessed text"):
            st.code(cleaned)