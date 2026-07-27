import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    @property
    def GROQ_API_KEY(self) -> str:
        return os.getenv("GROQ_API_KEY", "")

    @property
    def GEMINI_API_KEY(self) -> str:
        return os.getenv("GEMINI_API_KEY", "")

    @property
    def HF_API_KEY(self) -> str:
        return os.getenv("HF_API_KEY", "")

    @property
    def PDF_DATA_PATH(self) -> str | None:
        """Optional path to the folder containing PDF documents.
        If not set, the loader defaults to the relative "data" folder.
        """
        return os.getenv("PDF_DATA_PATH")

    @property 
    def MONGO_URI(self) -> str:
        return os.getenv("MONGO_URI", "mongodb://localhost:27017")

    @property
    def MONGO_URL(self) -> str:
        return os.getenv("MONGO_DB_URL", "mongodb://localhost:27017")

    @property
    def MONGO_DB_NAME(self) -> str:
        return os.getenv("MONGO_DB_NAME", "sports_rag")

    @property
    def DB_NAME(self) -> str:
        return self.MONGO_DB_NAME


settings = Settings()


class ModelConfig:

    def __init__(self, name, model):
        self.name = name
        self.model = model


FAST_MODELS = [
    ModelConfig(
        "GPT-OSS",
        "gpt-oss/120b"
    ),
    ModelConfig(
        "Groq",
        "groq/llama-3.3-70b-versatile"
    )
]


PRO_MODELS = [
    ModelConfig(
        "Groq",
        "groq/llama-3.3-70b-versatile"
    ),
    ModelConfig(
        "GPT-OSS",
        "gpt-oss/120b"
    )
]


def get_models(mode: str):
    if mode.lower() == "fast":
        return FAST_MODELS
    elif mode.lower() == "pro":
        return PRO_MODELS
    return None