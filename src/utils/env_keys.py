from enum import Enum

class EnvKeys(Enum):
    OPENSEARCH_PASSWORD = 'OPENSEARCH_INITIAL_ADMIN_PASSWORD'
    OPENSEARCH_USERNAME = 'OPENSEARCH_INITIAL_ADMIN_USERNAME'
    OPENSEARCH_PORT = 'OPENSEARCH_PORT'
    OPENSEARCH_INDEX_NAME = 'OPENSEARCH_INDEX_NAME'
    OPENSEARCH_HOST = 'OPENSEARCH_HOST'
    MONGO_HOST = 'MONGO_HOST'
    MONGO_DB = 'MONGO_DB'
    MONGO_CLUSTER = 'MONGO_CLUSTER'
    CLIENT_URL = 'CLIENT_URL'
    SERVER_URL = 'SERVER_URL'
    SECRET_KEY = 'SECRET_KEY'
    HASHING_ALGO = 'HASHING_ALGO'
    ACCESS_TOKEN_EXPIRATION = 'ACCESS_TOKEN_EXPIRE_MINUTES'
    SLOT_FILLER_MODEL = 'SLOT_FILLER_MODEL'
    CONTEXT_TRANSLATOR_MODEL = 'CONTEXT_TRANSLATOR_MODEL'
    FEATURE_EXTRACTION_MODEL = 'FEATURE_EXTRACTION_MODEL'
    RAG_BASED_BOOKING_MODEL = 'RAG_BASED_BOOKING_MODEL'
    INTENT_CLASSIFIER_MODEL = 'INTENT_CLASSIFIER_MODEL'
