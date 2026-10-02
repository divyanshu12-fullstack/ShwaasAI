from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class ConfigurationError(RuntimeError):
    pass


class Settings:
    def __init__(self) -> None:
        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.publishable_key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
        self.jwks_url = os.getenv("SUPABASE_JWKS_URL", "")
        # Only the analysis result writer uses this server-side key, after
        # verifying the caller and checking ownership with the caller's JWT.
        self.secret_key = os.getenv("SUPABASE_SECRET_KEY", "")
        # Normal profile/session reads still use the caller's token and RLS.
        # SESSION_POOLER_URL is reserved for server-side SQL migrations. API
        # requests use the Supabase Data API with the caller's JWT, not this URL.
        if not self.supabase_url.startswith("https://") or not self.publishable_key:
            raise ConfigurationError("SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY are required")


@lru_cache
def get_settings() -> Settings:
    return Settings()
