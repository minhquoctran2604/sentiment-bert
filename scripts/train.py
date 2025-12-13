
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_loader import load_and_prepare_data
from src.model import load_model
from src.trainer import create_trainer

def main():
    print("="*60)
    print("SENTIMENT ANALYSIS - DOMAIN ADAPTATION")
    print("Twitter RoBERTa 3-class")
    print("="*60)
    
    # Load configuration
    config = Config()
    
    # Load data
    train_dataset, eval_dataset, tokenizer = load_and_prepare_data(config)
    
    # Load model
    model = load_model(config)
    
    # Create trainer
    print("\nSetting up trainer...")
    trainer = create_trainer(model, train_dataset, eval_dataset, tokenizer, config)
    print("Trainer initialized")
    
    print(f"\nTraining configuration:")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Epochs: {config.training.num_epochs}")
    print(f"  - Batch size: {config.training.batch_size}")
    print(f"  - Warmup ratio: {config.training.warmup_ratio}")
    
    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING")
    print("="*60)
    
    results = trainer.evaluate()
    
    print("\nFINAL RESULTS:")
    print(f"  Accuracy: {results['eval_accuracy']:.2%}")
    print(f"  F1 Score: {results['eval_f1']:.4f}")
    print(f"  Loss: {results['eval_loss']:.4f}")
    
    # Save model
    print("\nSaving model...")
    model_path = config.training.models_dir / "best_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✓ Model saved to {model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE! 🎉")
    print("="*60)

if __name__ == "__main__":
    main()
