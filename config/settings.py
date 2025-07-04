from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LOGS_DIR: Path = BASE_DIR / 'logs'
    UPLOADS_DIR: Path = BASE_DIR / 'uploads'
    MODEL_PATH: Path = BASE_DIR / 'model' / 'trained_model.pth'
    VECTOR_PATH: Path = BASE_DIR / 'model' / 'vectorizer.pkl'
    CLASSES: Path = BASE_DIR / 'classes' / 'arget_names.pkl'

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings() 