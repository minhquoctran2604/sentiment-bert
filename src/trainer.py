"""
Training utilities
"""
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }

def create_trainer(model, train_dataset, eval_dataset, tokenizer, config):
    training_args = TrainingArguments(
        output_dir=str(config.training.outputs_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        num_train_epochs=config.training.num_epochs,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer
