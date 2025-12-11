import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from transformers import pipeline
from src.config import Config

# Page config
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎭",
    layout="centered"
)

# Load model
@st.cache_resource
def load_classifier():
    config = Config()
    model_path = config.training.models_dir / "best_model"
    return pipeline("sentiment-analysis", model=str(model_path))

# UI
st.title("🎭 Movie Review Sentiment Analysis")
st.markdown("**Powered by fine-tuned BERT (Domain Adaptation: SST-2 → IMDB)**")

st.divider()

# Input
text = st.text_area(
    "Enter a movie review:",
    placeholder="e.g., I loved this movie! The acting was superb...",
    height=150
)

# Analyze button
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if text.strip():
        with st.spinner("Analyzing..."):
            classifier = load_classifier()
            result = classifier(text)[0]
            
            # Display result
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if result['label'] == 'LABEL_1':
                    st.success("### 😊 POSITIVE")
                else:
                    st.error("### 😞 NEGATIVE")
            
            with col2:
                st.metric("Confidence", f"{result['score']:.1%}")
    else:
        st.warning("⚠️ Please enter some text first!")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This model uses **domain adaptation**:
    
    1. Start from SST-2 fine-tuned BERT
    2. Adapt to IMDB user reviews
    3. Achieve 89%+ accuracy
    
    **Tech Stack:**
    - Model: DistilBERT
    - Framework: Hugging Face Transformers
    - Dataset: IMDB
    """)

st.divider()
st.caption("Built with ❤️ using Streamlit & Hugging Face 🤗")
