from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def load_and_prepare_data(config):
    # ===== ĐỔI: SST5 thay vì IMDB =====
    dataset: DatasetDict = load_dataset("SetFit/sst5", cache_dir=str(config.training.data_dir))  # type: ignore
    print(f"Train: {len(dataset['train']):,}, Validation: {len(dataset['validation']):,}")
    
    # ===== THÊM: Map 5 → 3 classes =====
    def map_to_3_classes(example):
        label = example['label']
        if label <= 1:      # Very neg, Neg
            example['label'] = 0
        elif label == 2:    # Neutral
            example['label'] = 1
        else:               # Pos, Very pos
            example['label'] = 2
        return example
    
    dataset = dataset.map(map_to_3_classes)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],  # SST5 dùng 'text'
            padding="max_length",
            truncation=True,
            max_length=config.model.max_length
        )
    
    tokenized = dataset.map(tokenize_function, batched=True)
    
    train_dataset = tokenized["train"].shuffle(
        seed=config.training.seed
    ).select(range(config.training.train_size))
    
    test_dataset = tokenized["validation"].shuffle(
        seed=config.training.seed
    ).select(range(config.training.test_size))
    
    print(f"Train subset: {len(train_dataset)}")
    print(f"Test subset: {len(test_dataset)}")
    
    return train_dataset, test_dataset, tokenizer
