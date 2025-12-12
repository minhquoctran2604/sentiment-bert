import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # type: ignore
from transformers import pipeline
from src.config import Config

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="centered")

@st.cache_resource
def load_classifier():
    config = Config()
    model_path = config.training.models_dir / "best_model"
    return pipeline("sentiment-analysis", model=str(model_path))

st.title("🎭 Movie Review Sentiment Analysis")
st.markdown("**Powered by fine-tuned RoBERTa (3-class)**")
st.divider()

text = st.text_area("Enter a movie review:", placeholder="e.g., I loved this movie!", height=150)

if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if text.strip():
        with st.spinner("Analyzing..."):
            classifier = load_classifier()
            result = classifier(text)[0]  # type: ignore
            label: str = result['label']  # type: ignore
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if label == 'POSITIVE':
                    st.success("### 😊 POSITIVE")
                elif label == 'NEGATIVE':
                    st.error("### 😞 NEGATIVE")
                else:
                    st.info("### 😐 NEUTRAL")
            with col2:
                st.metric("Confidence", f"{result['score']:.1%}")  # type: ignore
    else:
        st.warning("⚠️ Please enter some text first!")

with st.sidebar:
    st.header("About")
    st.markdown("""
    **3-Class Sentiment:**
    - 😊 POSITIVE
    - 😐 NEUTRAL  
    - 😞 NEGATIVE
    
    Model: Twitter RoBERTa + SST-5
    """)

st.divider()
st.caption("Built with ❤️ using Streamlit & Hugging Face 🤗")
