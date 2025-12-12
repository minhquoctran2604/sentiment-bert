import sys
from pathlib import Path
from typing import Any, Dict, List
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # type: ignore
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
st.markdown("**Powered by fine-tuned BERT (3-class: Positive / Neutral / Negative)**")

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
            results: List[Dict[str, Any]] = classifier(text)  # type: ignore
            result = results[0]
            label: str = result['label']
            score: float = result['score']
            
            # Display result
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if label == 'POSITIVE':
                    st.success("### 😊 POSITIVE")
                elif label == 'NEGATIVE':
                    st.error("### 😞 NEGATIVE")
                elif label == 'NEUTRAL':
                    st.info("### 😐 NEUTRAL")
                else:
                    st.warning(f"### {label}")
            
            with col2:
                st.metric("Confidence", f"{score:.1%}")
            
            # Explanation
            if label == 'NEUTRAL':
                st.info("ℹ️ This review has mixed or neutral sentiment.")
    else:
        st.warning("⚠️ Please enter some text first!")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This model uses **3-class sentiment**:
    
    - 😊 **POSITIVE**: Very positive to positive
    - 😐 **NEUTRAL**: Mixed/neutral feelings
    - 😞 **NEGATIVE**: Negative to very negative
    
    **Tech Stack:**
    - Model: DistilBERT
    - Framework: Hugging Face Transformers
    - Dataset: SST (Stanford Sentiment Treebank)
    """)

st.divider()
st.caption("Built with ❤️ using Streamlit & Hugging Face 🤗")
