from transformers import AutoModelForSequenceClassification

def load_model(config):
    print(f"\nLoading {config.model.name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name,
        num_labels=config.model.num_labels,
        id2label=config.training.id2label,
        label2id=config.training.label2id
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {num_params:,} parameters")
    
    return model
