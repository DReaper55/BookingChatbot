from enum import Enum

class AssetPaths(Enum):
    # Models
    T5_MODEL = 'models/t5_model'
    T5_BOOKING_MODEL = 'models/t5_booking_model'
    T5_INTENT_CLASSIFIER_MODEL = 'models/t5_intent_classifier_model'
    T5_SLOT_EXTRACTION_MODEL = 'models/t5_slot_extraction_model'
    T5_MULTITASK_BOOKING_MODEL = 'models/t5_multitask_booking_model'

    # Weights

    # Transformers

    # Dataset
    TRAINING_DATASET = 'data/processed/train.json'
    VALIDATION_DATASET = 'data/processed/dev.json'
    TEST_DATASET = 'data/processed/test.json'

    RAW_SYNTHETIC_DATASET = 'data/raw/synthetic.txt'
    RAW_RAG_DATASET = 'data/raw/rag_dataset.json'
    PROCESSED_RAG_DATASET = 'data/processed/rag_dataset.json'
    # SYNTHETIC_DATASET = 'data/processed/synthetic.json'
    SYNTHETIC_DATASET = 'data/processed/modified_synthetic.json'

