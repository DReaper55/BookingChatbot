from enum import Enum

class AssetPaths(Enum):
    # Models
    T5_BOOKING_MODEL = 'models/t5_booking_model'
    T5_INTENT_CLASSIFIER_MODEL = 'models/t5_intent_classifier_model'
    T5_SLOT_EXTRACTION_MODEL = 'models/t5_slot_extraction_model'
    T5_MULTITASK_BOOKING_MODEL = 'models/t5_multitask_booking_model'
    T5_MULTITASK_FEATURE_EXTRACTION_MODEL = 'models/t5_multitask_feature_extraction_model'
    T5_CONTEXT_TRANSLATOR_MODEL = 'models/t5_context_translator_model'
    T5_SLOT_FILLER_MODEL = 'models/t5_slot_filler_model'

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
    RAW_CONTEXT_TRANSLATOR_DATASET = 'data/raw/context_translator_dataset.json'
    CONTEXT_TRANSLATOR_DATASET = 'data/processed/context_translator_dataset.json'

    FEATURE_EXTRACTION_DATASET = 'data/processed/feature_extraction_dataset.json'
    RAW_FEATURE_EXTRACTION_DATASET = 'data/raw/feature_extraction_dataset.json'

    RAW_SLOT_FILLER_DATASET = 'data/raw/slot_filling_dataset.json'
    SLOT_FILLER_DATASET = 'data/processed/slot_filler_dataset.json'
