from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "SIE LLM Search Backend"
    hf_token: str = ""
    hf_cache_dir: Path = Path("data") / "hf-cache"

    # SQLite DB in data/sqlite/
    sqlite_path: Path = Path("data") / "sqlite" / "sie.db"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-3.1-pro-preview"
    llm_max_parallel: int = 20

    # Superlinked Inference Engine
    sie_api_key: str = ""
    sie_api_endpoint: str = ""
    sie_gpu_profile: str = "l4-spot"
    sie_embed_batch_size: int = 32
    sie_embed_model: str = "NovaSearch/stella_en_400M_v5"

    # Chroma
    chroma_path: Path = Path("data") / "chroma"

    @property
    def database_url(self) -> str:
        db_path = self.sqlite_path.resolve()
        return f"sqlite:///{db_path}"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
