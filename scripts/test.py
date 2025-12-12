"""Test trained model"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import pipeline
from src.config import Config

def main():
    config = Config()
    model_path = config.training.models_dir / "best_model"
    
    print("Loading model...")
    classifier = pipeline("sentiment-analysis", model=str(model_path))
    print("✓ Model loaded!\n")
    
    tests = [
        "I absolutely loved this movie! The acting was superb.",
        "This was a complete waste of time. Boring and predictable.",
        "Great cinematography but the story was a bit slow.",
        "Best film I've seen this year! Highly recommend!",
        "Terrible acting, poor script, don't waste your money.",
    ]
    
    print("="*60)
    print("TESTING MODEL")
    print("="*60)
    
    emoji_map = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}
    
    for i, text in enumerate(tests, 1):
        result = classifier(text)[0]  # type: ignore
        label: str = result['label']  # type: ignore
        emoji = emoji_map.get(label, "❓")
        
        print(f"\n[{i}] {text}")
        print(f"→ {emoji} {label} (confidence: {result['score']:.1%})")  # type: ignore
    
    print("\n" + "="*60)
    print("✓ Model works!")
    print("="*60)

if __name__ == "__main__":
    main()
