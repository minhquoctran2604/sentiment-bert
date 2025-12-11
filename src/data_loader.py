from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_prepare_data(config):
    
    dataset = load_dataset("imdb", cache_dir=str(config.training.data_dir))
    print(f"Train: {len(dataset['train']):,}, Test: {len(dataset['test']):,}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    # tokenization 
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.model.max_length
        )
    
    tokenized = dataset.map(tokenize_function, batched=True)
    
    # Create subsets
    train_dataset = tokenized["train"].shuffle(
        seed=config.training.seed
    ).select(range(config.training.train_size))
    
    test_dataset = tokenized["test"].shuffle(
        seed=config.training.seed
    ).select(range(config.training.test_size))
    
    print(f" Train subset: {len(train_dataset)}")
    print(f" Test subset: {len(test_dataset)}")
    
    return train_dataset, test_dataset, tokenizer
