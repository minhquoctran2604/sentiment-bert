"""
Test trained model
"""
import sys
from pathlib import Path
from typing import Any, Dict, List
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import pipeline
from src.config import Config

def main():
    config = Config()
    model_path = config.training.models_dir / "best_model"
    
    print("Loading model...")
    classifier = pipeline("sentiment-analysis", model=str(model_path))
    print("✓ Model loaded!\n")
    
    # Test cases with expected labels
    tests: List[tuple[str, str]] = [
        ("I absolutely loved this movie! The acting was superb.", "POSITIVE"),
        ("This was a complete waste of time. Boring and predictable.", "NEGATIVE"),
        ("Great cinematography but the story was a bit slow.", "NEUTRAL"),
        ("It was okay, nothing special.", "NEUTRAL"),
        ("Best film I've seen this year! Highly recommend!", "POSITIVE"),
        ("Terrible acting, poor script, don't waste your money.", "NEGATIVE"),
    ]
    
    # Emoji map for 3-class
    emoji_map: Dict[str, str] = {
        "POSITIVE": "😊",
        "NEGATIVE": "😞",
        "NEUTRAL": "😐"
    }
    
    print("="*60)
    print("TESTING MODEL (3-CLASS: POSITIVE / NEUTRAL / NEGATIVE)")
    print("="*60)
    
    correct = 0
    for i, (text, expected) in enumerate(tests, 1):
        results: List[Dict[str, Any]] = classifier(text)  # type: ignore
        result = results[0]
        label: str = result['label']
        score: float = result['score']
        emoji = emoji_map.get(label, "❓")
        
        match = "✓" if label == expected else "✗"
        if label == expected:
            correct += 1
        
        print(f"\n[{i}] {text}")
        print(f"→ {emoji} {label} (confidence: {score:.1%}) {match}")
        print(f"   Expected: {expected}")
    
    print("\n" + "="*60)
    print(f"✓ Accuracy: {correct}/{len(tests)} ({correct/len(tests)*100:.0f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
