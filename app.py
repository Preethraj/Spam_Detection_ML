import streamlit as st
import joblib, re, string

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/spam_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+','', text)
    text = text.translate(str.maketrans('','',string.punctuation))
    return text

model, vectorizer = load_artifacts()

st.title("📩 SmartSMS Shield — Spam Detector")
st.write("Enter a message and click Predict.")

msg = st.text_area("Message to check", height=150)

if st.button("Predict"):
    if not msg.strip():
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(msg)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.error("🚨 Spam detected")
        else:
            st.success("✅ Not spam")
