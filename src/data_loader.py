from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def load_and_prepare_data(config):    
    print("Loading SST-5 dataset...")
    
    # Use SetFit/sst5 - the modern HuggingFace version of SST-5
    # Labels: 0=very negative, 1=negative, 2=neutral, 3=positive, 4=very positive
    dataset: DatasetDict = load_dataset("SetFit/sst5", cache_dir=str(config.training.data_dir))  # type: ignore
    
    # Map 5 labels [0,1,2,3,4] to 3 classes
    def map_to_3_classes(example):
        label = example['label']
        if label <= 1:      # Very negative (0), Negative (1)
            new_label = 0   # NEGATIVE
        elif label == 2:    # Neutral
            new_label = 1   # NEUTRAL
        else:               # Positive (3), Very positive (4)
            new_label = 2   # POSITIVE
        return {'label': new_label, 'text': example['text']}
    
    dataset = dataset.map(map_to_3_classes)
    
    train_data = dataset['train']
    val_data = dataset['validation']
    
    print(f"✓ Loaded {len(train_data):,} train samples")
    print(f"✓ Loaded {len(val_data):,} validation samples")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    # Tokenization - SST-5 uses 'text' field
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
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
