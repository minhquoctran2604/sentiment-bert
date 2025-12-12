from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def load_and_prepare_data(config):
    """Load SST dataset and map to 3 classes"""
    
    print("Loading SST dataset...")
    
    # Load SST-5 dataset (5-class sentiment)
    dataset: DatasetDict = load_dataset("sst", "default", cache_dir=str(config.training.data_dir))  # type: ignore
    
    # Map 5 labels [0,1,2,3,4] to 3 classes
    def map_to_3_classes(example):
        label = example['label']
        if label <= 1:      # Very negative (0), Negative (1)
            example['label'] = 0  # NEGATIVE
        elif label == 2:    # Neutral
            example['label'] = 1  # NEUTRAL
        else:               # Positive (3), Very positive (4)
            example['label'] = 2  # POSITIVE
        return example
    
    dataset = dataset.map(map_to_3_classes)
    
    train_data = dataset['train']
    val_data = dataset['validation']
    
    print(f"✓ Loaded {len(train_data):,} train samples")
    print(f"✓ Loaded {len(val_data):,} validation samples")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    # Tokenization - SST uses 'sentence' field
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config.model.max_length
        )
    
    tokenized = dataset.map(tokenize_function, batched=True)
    
    # Create subsets
    train_dataset = tokenized["train"].shuffle(seed=config.training.seed).select(
        range(min(config.training.train_size, len(tokenized["train"])))
    )
    
    test_dataset = tokenized["validation"].shuffle(seed=config.training.seed).select(
        range(min(config.training.test_size, len(tokenized["validation"])))
    )
    
    print(f"✓ Train subset: {len(train_dataset)}")
    print(f"✓ Test subset: {len(test_dataset)}")
    
    return train_dataset, test_dataset, tokenizer
