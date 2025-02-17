from enum import Enum

class AssetPaths(Enum):
    # Models
    T5_MODEL = 'models/t5_booking_model'

    # Weights

    # Transformers

    # Dataset
    TRAINING_DATASET = 'data/processed/train.json'
    VALIDATION_DATASET = 'data/processed/dev.json'
    TEST_DATASET = 'data/processed/test.json'
