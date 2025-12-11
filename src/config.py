
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class ModelConfig:

    name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    num_labels: int = 2
    max_length: int = 512

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    train_size: int = 1000
    test_size: int = 200
    seed: int = 1
    
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    
    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
