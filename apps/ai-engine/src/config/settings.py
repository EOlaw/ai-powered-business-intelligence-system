"""
InsightSerenity AI Engine — Configuration
==========================================
Central, Pydantic-validated settings loaded from environment variables and
optional .env files. All tuneable constants live here; nothing is hard-coded
inside business logic.

Usage:
    from src.config.settings import settings
    print(settings.device)          # "cuda" | "mps" | "cpu"
    print(settings.crawler.max_depth)
"""

import os
import multiprocessing
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Environment(str, Enum):
    """Deployment environment — controls log verbosity and safety guards."""
    DEVELOPMENT = "development"
    STAGING     = "staging"
    PRODUCTION  = "production"
    TEST        = "test"


class LogLevel(str, Enum):
    DEBUG    = "DEBUG"
    INFO     = "INFO"
    WARNING  = "WARNING"
    ERROR    = "ERROR"
    CRITICAL = "CRITICAL"


class Device(str, Enum):
    """Compute device for PyTorch tensors."""
    CUDA = "cuda"
    MPS  = "mps"      # Apple Silicon
    CPU  = "cpu"
    AUTO = "auto"     # Resolved at runtime


# ─────────────────────────────────────────────────────────────────────────────
# Sub-settings groups (nested Pydantic models)
# ─────────────────────────────────────────────────────────────────────────────

class StoragePaths(BaseSettings):
    """
    Filesystem paths for all persistent artefacts.
    All paths are resolved to absolute and created on first access.
    """
    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    # Root storage directory (everything lives under here)
    root: Path = Field(
        default=Path("storage"),
        description="Root directory for all AI engine storage artefacts.",
    )

    @property
    def models(self) -> Path:
        """Directory where trained model weight files (.pt/.safetensors) are saved."""
        return self.root / "models"

    @property
    def checkpoints(self) -> Path:
        """Mid-training checkpoint files — used to resume interrupted runs."""
        return self.root / "checkpoints"

    @property
    def tokenizers(self) -> Path:
        """Saved tokenizer vocab and merge files."""
        return self.root / "tokenizers"

    @property
    def datasets(self) -> Path:
        """Processed, cleaned, and deduplicated training corpus files."""
        return self.root / "datasets"

    @property
    def logs(self) -> Path:
        """Training logs and experiment tracking output."""
        return self.root / "logs"

    def ensure_all(self) -> None:
        """Create all storage subdirectories if they do not yet exist."""
        for path in [
            self.root, self.models, self.checkpoints,
            self.tokenizers, self.datasets, self.logs,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class CrawlerSettings(BaseSettings):
    """
    Controls the async web crawler's behaviour, politeness limits,
    and what content is accepted or rejected.
    """
    model_config = SettingsConfigDict(env_prefix="CRAWLER_")

    # Maximum link-traversal depth from the seed URL
    max_depth: int = Field(default=3, ge=1, le=10)

    # Total page cap per crawl run — prevents runaway crawls
    max_pages: int = Field(default=10_000, ge=1)

    # Concurrent HTTP connections across all domains
    max_concurrent: int = Field(default=50, ge=1, le=500)

    # Seconds to wait between requests to the SAME domain (politeness)
    delay_per_domain: float = Field(default=1.0, ge=0.1)

    # HTTP request timeout in seconds
    request_timeout: int = Field(default=30, ge=5, le=120)

    # User-Agent string sent in every request
    user_agent: str = Field(
        default="InsightSerenityBot/1.0 (+https://insightserenity.com/bot)"
    )

    # Whether to respect robots.txt exclusion rules
    respect_robots_txt: bool = Field(default=True)

    # Only crawl these content types (reject binary, media, etc.)
    allowed_content_types: List[str] = Field(
        default=["text/html", "text/plain", "application/xhtml+xml"]
    )

    # Reject URLs whose path matches any of these substrings
    excluded_url_patterns: List[str] = Field(
        default=[
            "/login", "/signup", "/register", "/cart", "/checkout",
            ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".zip",
        ]
    )

    # Max size of a single response body in bytes (skip larger pages)
    max_response_bytes: int = Field(default=5 * 1024 * 1024)  # 5 MB


class PreprocessingSettings(BaseSettings):
    """Controls text cleaning and normalisation behaviour."""
    model_config = SettingsConfigDict(env_prefix="PREPROCESSING_")

    # Discard documents shorter than this many whitespace-separated tokens
    min_tokens: int = Field(default=50)

    # Discard documents longer than this (avoids gigantic single-document noise)
    max_tokens: int = Field(default=100_000)

    # Minimum ratio of alphabetic characters to total characters in a document
    min_alpha_ratio: float = Field(default=0.6, ge=0.0, le=1.0)

    # Convert all text to lowercase before saving
    lowercase: bool = Field(default=False)

    # Collapse multiple consecutive newlines into one
    collapse_newlines: bool = Field(default=True)

    # Strip HTML entities (e.g. &amp; → &)
    decode_html_entities: bool = Field(default=True)


class DeduplicationSettings(BaseSettings):
    """Controls how near and exact duplicates are detected and removed."""
    model_config = SettingsConfigDict(env_prefix="DEDUP_")

    # Number of MinHash permutations — higher = more accurate, slower
    minhash_num_perm: int = Field(default=128, ge=32, le=512)

    # Jaccard similarity threshold above which two docs are considered duplicates
    minhash_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # Size of the n-gram shingles used for MinHash similarity
    minhash_ngram_size: int = Field(default=5, ge=1, le=10)

    # Batch size for bulk MinHash computation
    batch_size: int = Field(default=10_000, ge=100)


class QualityFilterSettings(BaseSettings):
    """Thresholds for the quality-filtering stage of the data pipeline."""
    model_config = SettingsConfigDict(env_prefix="QUALITY_")

    # Only keep documents whose detected language matches this list
    allowed_languages: List[str] = Field(default=["en"])

    # Minimum language detection confidence (0–1)
    min_language_confidence: float = Field(default=0.9, ge=0.0, le=1.0)

    # Maximum fraction of a document that may be repeated n-grams
    max_repetition_ratio: float = Field(default=0.3, ge=0.0, le=1.0)

    # Maximum fraction of lines that are bullet points / short fragments
    max_bullet_ratio: float = Field(default=0.9, ge=0.0, le=1.0)

    # Perplexity threshold — documents above this are considered low-quality
    # (only applied when a reference LM is available)
    max_perplexity: Optional[float] = Field(default=None)


class TokenizerSettings(BaseSettings):
    """Defaults for tokenizer training."""
    model_config = SettingsConfigDict(env_prefix="TOKENIZER_")

    # Target vocabulary size (shared across BPE/WordPiece)
    vocab_size: int = Field(default=32_000, ge=1_000, le=256_000)

    # Minimum token frequency in corpus to be included in the vocabulary
    min_frequency: int = Field(default=2, ge=1)

    # Maximum input length (in tokens) the tokenizer will produce
    max_length: int = Field(default=2048, ge=64)

    # Whether to add a beginning-of-sequence token automatically
    add_bos_token: bool = Field(default=True)

    # Whether to add an end-of-sequence token automatically
    add_eos_token: bool = Field(default=True)


class TrainingSettings(BaseSettings):
    """Default hyper-parameters and infrastructure settings for training runs."""
    model_config = SettingsConfigDict(env_prefix="TRAINING_")

    # --- Compute ---
    # Number of CPU workers for DataLoader
    num_workers: int = Field(
        default=min(4, multiprocessing.cpu_count()),
        ge=0,
    )

    # Whether to pin memory for faster GPU transfers
    pin_memory: bool = Field(default=True)

    # Whether to use PyTorch automatic mixed precision (fp16/bf16)
    use_amp: bool = Field(default=True)

    # AMP dtype: "float16" or "bfloat16"
    amp_dtype: str = Field(default="bfloat16")

    # --- Optimisation defaults ---
    learning_rate: float   = Field(default=3e-4, gt=0)
    weight_decay: float    = Field(default=0.1,  ge=0)
    gradient_clip: float   = Field(default=1.0,  gt=0)
    warmup_steps: int      = Field(default=1_000, ge=0)
    batch_size: int        = Field(default=32,   ge=1)
    gradient_accumulation: int = Field(default=1, ge=1)

    # --- Checkpointing ---
    save_every_n_steps: int = Field(default=1_000, ge=1)
    keep_last_n_checkpoints: int = Field(default=5,   ge=1)

    # --- Evaluation ---
    eval_every_n_steps: int = Field(default=500, ge=1)
    eval_max_batches: int   = Field(default=100, ge=1)


class ServingSettings(BaseSettings):
    """FastAPI inference server configuration."""
    model_config = SettingsConfigDict(env_prefix="SERVING_")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001, ge=1024, le=65535)

    # Maximum number of tokens to generate in a single completion
    max_new_tokens: int = Field(default=2048, ge=1)

    # How many requests may be in-flight simultaneously
    max_concurrent_requests: int = Field(default=32, ge=1)

    # Internal shared secret used by the Node.js gateway for service-to-service auth
    internal_api_secret: str = Field(default="change-me-in-production")

    # Stream responses token-by-token via SSE by default
    default_stream: bool = Field(default=True)


# ─────────────────────────────────────────────────────────────────────────────
# Root Settings
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Master settings object. Instantiated once at import time and shared
    across the entire ai-engine application via `settings`.

    Values are loaded in this priority order (highest wins):
    1. Actual environment variables
    2. Variables in a .env file at the repo root
    3. Field defaults defined below
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",         # Silently ignore unknown env vars
    )

    # ── Top-level ─────────────────────────────────────────────────────────────
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    log_level: LogLevel      = Field(default=LogLevel.INFO)
    debug: bool              = Field(default=False)
    project_name: str        = Field(default="InsightSerenity AI Engine")
    version: str             = Field(default="1.0.0")

    # ── Device ────────────────────────────────────────────────────────────────
    device: Device = Field(
        default=Device.AUTO,
        description="Compute device. AUTO detects CUDA > MPS > CPU at runtime.",
    )

    # ── Sub-settings (each reads its own prefixed env vars) ──────────────────
    storage:        StoragePaths          = Field(default_factory=StoragePaths)
    crawler:        CrawlerSettings       = Field(default_factory=CrawlerSettings)
    preprocessing:  PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    deduplication:  DeduplicationSettings = Field(default_factory=DeduplicationSettings)
    quality_filter: QualityFilterSettings = Field(default_factory=QualityFilterSettings)
    tokenizer:      TokenizerSettings     = Field(default_factory=TokenizerSettings)
    training:       TrainingSettings      = Field(default_factory=TrainingSettings)
    serving:        ServingSettings       = Field(default_factory=ServingSettings)

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("device", mode="before")
    @classmethod
    def resolve_device(cls, v: str | Device) -> str:
        """
        If device is AUTO, detect the best available hardware at runtime.
        Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
        """
        value = v.value if isinstance(v, Device) else str(v).lower()
        if value == Device.AUTO.value:
            if torch.cuda.is_available():
                return Device.CUDA.value
            if torch.backends.mps.is_available():
                return Device.MPS.value
            return Device.CPU.value
        return v

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_flag(cls, v: object) -> object:
        """Accept common non-boolean DEBUG values used by shells/build tools."""
        if isinstance(v, str):
            value = v.strip().lower()
            if value in {"1", "true", "yes", "y", "on", "debug", "development"}:
                return True
            if value in {"0", "false", "no", "n", "off", "release", "production", ""}:
                return False
        return v

    @model_validator(mode="after")
    def enforce_production_guards(self) -> "Settings":
        """
        In production, refuse to start with obviously insecure defaults.
        This prevents misconfigured deployments from running silently.
        """
        if self.environment == Environment.PRODUCTION:
            if self.serving.internal_api_secret == "change-me-in-production":
                raise ValueError(
                    "SERVING_INTERNAL_API_SECRET must be set to a secure value in production."
                )
            if self.debug:
                raise ValueError("DEBUG must be False in production.")
        return self

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def torch_device(self) -> torch.device:
        """Returns a torch.device object ready to pass to .to() calls."""
        return torch.device(self.device.value)

    @property
    def amp_dtype_torch(self) -> torch.dtype:
        """Returns the torch dtype corresponding to the configured AMP dtype."""
        mapping = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        return mapping.get(self.training.amp_dtype, torch.bfloat16)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

# Instantiate once. All modules import this object rather than re-parsing env.
settings = Settings()

# Ensure storage directories exist as soon as settings are loaded
settings.storage.ensure_all()
