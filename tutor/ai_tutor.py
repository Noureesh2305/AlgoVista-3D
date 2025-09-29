# tutor/ai_tutor.py

import streamlit as st
from gpt4all import GPT4All
import os

# Absolute path to your GPT4All model file
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "tinyllama-1.1b-1t-openorca.Q4_K_M.gguf")
)

@st.cache_resource
def load_model():
    """Load GPT4All model fully offline."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    try:
        # Pass model path directly, disable downloads
        return GPT4All(MODEL_PATH, allow_download=False)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

def render():
    st.subheader("🤖 AI Tutor – Offline ML Doubt Solver")
    st.markdown(
        "Ask anything about Machine Learning and I’ll answer it using a **local AI model** – no internet required."
    )

    user_input = st.text_input("💬 Ask your question:")

    if user_input:
        model = load_model()
        with st.spinner("💡 Thinking..."):
            try:
                response = model.generate(
                    f"You are an expert Machine Learning tutor. Answer the following question in simple, clear terms:\n{user_input}",
                    max_tokens=300
                )
                st.success(f"📘 {response}")
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
