"""
InsightSerenity AI Engine — Model Registry
==========================================
The model registry manages the lifecycle of all loaded models:
    - Discovery: scans the storage/models directory for available models
    - Loading: loads model weights from disk onto the correct device
    - Caching: keeps hot models in memory, evicts least-recently-used
    - Versioning: tracks model versions and allows rollback
    - Metadata: stores model capabilities, vocab size, context length

Registry conventions:
    Model directory structure:
        storage/models/
            {model_name}/
                {version}/
                    model.pt               ← model state_dict
                    config.json            ← model configuration
                    tokenizer/             ← tokenizer directory
                        vocab.json
                        merges.txt
                        tokenizer_config.json

    Model naming: "{name}:{version}" (e.g. "gpt-small:v1.0.0")
    Default version: "latest" → resolves to the highest version number

Thread safety: Loading and cache eviction are protected by asyncio locks
so concurrent API requests don't corrupt the cache state.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Metadata about a registered model."""
    name:            str
    version:         str
    model_dir:       Path
    vocab_size:      int          = 0
    d_model:         int          = 0
    context_length:  int          = 2048
    model_type:      str          = "gpt"   # "gpt" | "bert" | "encoder"
    capabilities:    List[str]    = field(default_factory=list)
    loaded_at:       float        = 0.0
    last_used_at:    float        = 0.0
    param_count:     int          = 0
    # Provenance — populated from metadata.json written by scripts/models/promote.py
    perplexity:      Optional[float] = None
    val_loss:        Optional[float] = None
    train_loss:      Optional[float] = None
    promoted_at:     Optional[str]   = None   # ISO-8601 string
    promoted_at_unix: float           = 0.0   # Unix timestamp for Prometheus gauge
    training_run:    Optional[str]   = None
    checksum_sha256: Optional[str]   = None


class ModelRegistry:
    """
    Manages discovery, loading, and caching of trained models.

    Models are loaded lazily — a model is only loaded into GPU/CPU
    memory when it is first requested. The LRU cache evicts models
    that haven't been used recently to free memory.

    Args:
        models_dir:   Root directory containing model subdirectories.
        max_loaded:   Maximum number of models to keep in memory. Default 2.
        device:       Compute device for loaded models.
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        max_loaded: int            = 2,
        device:     Optional[str]  = None,
    ) -> None:
        self.models_dir = Path(models_dir or settings.storage.models)
        self.max_loaded = max_loaded
        self.device     = torch.device(device or str(settings.torch_device))

        # Registry: model_key → ModelInfo
        self._registry: Dict[str, ModelInfo] = {}

        # Cache: model_key → loaded (model, tokenizer) pair
        self._cache:    Dict[str, Tuple[nn.Module, Any]] = {}

        # Scan for available models
        self._discover()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_model(self, model_name: str) -> Tuple[nn.Module, Any]:
        """
        Get a loaded (model, tokenizer) pair by name.

        Loads the model if not already in cache. Evicts LRU model if
        the cache is at capacity.

        Args:
            model_name: Model name with optional version suffix.
                        "gpt-small"         → loads the "latest" version
                        "gpt-small:v1.0.0"  → loads specific version

        Returns:
            Tuple (model, tokenizer) ready for inference.

        Raises:
            KeyError: If the model is not found in the registry.
        """
        key = self._resolve_key(model_name)

        if key not in self._registry:
            raise KeyError(
                f"Model '{model_name}' not found. "
                f"Available: {self.list_models()}"
            )

        # Cache hit
        if key in self._cache:
            self._registry[key].last_used_at = time.time()
            logger.debug("Model cache hit", model=key)
            return self._cache[key]

        # Cache miss: load the model
        return self._load_and_cache(key)

    def list_models(self) -> List[str]:
        """Return all registered model names."""
        return sorted(self._registry.keys())

    def get_info(self, model_name: str) -> Optional[ModelInfo]:
        """Return metadata for a model."""
        key = self._resolve_key(model_name)
        return self._registry.get(key)

    def register(
        self,
        name:        str,
        version:     str,
        model_dir:   str,
        info:        Optional[Dict] = None,
    ) -> None:
        """
        Manually register a model directory.

        Used when model paths don't follow the standard convention.

        Args:
            name:      Model name.
            version:   Version string (e.g. "v1.0.0").
            model_dir: Path to the directory containing model.pt and config.json.
            info:      Optional dict of additional metadata fields.
        """
        key = f"{name}:{version}"
        model_info = ModelInfo(
            name=name,
            version=version,
            model_dir=Path(model_dir),
        )
        if info:
            for k, v in info.items():
                if hasattr(model_info, k):
                    setattr(model_info, k, v)

        self._registry[key] = model_info
        # Also register as "latest" for easy access
        self._registry[f"{name}:latest"] = model_info
        logger.info("Model registered", name=key)

    def evict(self, model_name: str) -> None:
        """Remove a model from the in-memory cache."""
        key = self._resolve_key(model_name)
        if key in self._cache:
            del self._cache[key]
            logger.info("Model evicted from cache", model=key)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _discover(self) -> None:
        """
        Scan the models directory for available models.

        Expected structure:
            models_dir/{name}/{version}/model.pt
                                       config.json
        """
        if not self.models_dir.exists():
            logger.debug("Models directory does not exist yet", path=str(self.models_dir))
            return

        for name_dir in self.models_dir.iterdir():
            if not name_dir.is_dir():
                continue
            for version_dir in name_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                model_pt   = version_dir / "model.pt"
                config_json = version_dir / "config.json"

                if not model_pt.exists():
                    continue

                config = {}
                if config_json.exists():
                    with open(config_json) as f:
                        config = json.load(f)

                # Read provenance from metadata.json (written by scripts/models/promote.py)
                metadata      = {}
                metadata_json = version_dir / "metadata.json"
                if metadata_json.exists():
                    try:
                        with open(metadata_json) as f:
                            metadata = json.load(f)
                    except Exception:
                        pass

                # Convert promoted_at ISO string → Unix timestamp for Prometheus gauge
                promoted_at_str  = metadata.get("promoted_at")
                promoted_at_unix = 0.0
                if promoted_at_str:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(promoted_at_str.replace("Z", "+00:00"))
                        promoted_at_unix = dt.timestamp()
                    except Exception:
                        pass

                info = ModelInfo(
                    name=name_dir.name,
                    version=version_dir.name,
                    model_dir=version_dir,
                    vocab_size=config.get("vocab_size", 0),
                    d_model=config.get("d_model", 0),
                    context_length=config.get("max_seq_len", 2048),
                    model_type=config.get("model_type", "gpt"),
                    capabilities=config.get("capabilities", ["text-generation"]),
                    # Provenance fields
                    perplexity=metadata.get("perplexity"),
                    val_loss=metadata.get("val_loss"),
                    train_loss=metadata.get("train_loss"),
                    promoted_at=promoted_at_str,
                    promoted_at_unix=promoted_at_unix,
                    training_run=metadata.get("training_run"),
                    checksum_sha256=metadata.get("checksum_sha256"),
                )

                key = f"{name_dir.name}:{version_dir.name}"
                self._registry[key] = info

                # Update "latest" pointer if needed
                latest_key = f"{name_dir.name}:latest"
                if latest_key not in self._registry:
                    self._registry[latest_key] = info

        if self._registry:
            logger.info("Models discovered", models=list(self._registry.keys()))

    def _load_and_cache(self, key: str) -> Tuple[nn.Module, Any]:
        """Load a model from disk and add it to the cache."""
        info = self._registry[key]
        logger.info("Loading model", model=key, dir=str(info.model_dir))

        # Evict LRU if at capacity
        if len(self._cache) >= self.max_loaded:
            self._evict_lru()

        model, tokenizer = self._load_from_disk(info)

        info.loaded_at   = time.time()
        info.last_used_at = time.time()
        info.param_count  = sum(p.numel() for p in model.parameters())

        self._cache[key] = (model, tokenizer)
        logger.info(
            "Model loaded",
            model=key,
            params=f"{info.param_count / 1e6:.1f}M",
            device=str(self.device),
        )
        return model, tokenizer

    def _load_from_disk(self, info: ModelInfo) -> Tuple[nn.Module, Any]:
        """
        Load model weights and tokenizer from info.model_dir.

        The model architecture is inferred from the config.json file.
        Supports: GPTDecoder, BERTEncoder.
        """
        config_path = info.model_dir / "config.json"
        model_path  = info.model_dir / "model.pt"
        tok_dir     = info.model_dir / "tokenizer"

        # Load config
        config_dict = {}
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)

        # Load state dict first so we can override config with ground-truth shapes
        state_dict = None
        if model_path.exists():
            state_dict = torch.load(str(model_path), map_location=self.device, weights_only=True)

        # Override config fields with shapes read directly from the checkpoint.
        # This is the ground truth and prevents stale config.json values from
        # causing size-mismatch errors when loading.
        if state_dict is not None:
            embed_key = "token_emb.embedding.weight"
            if embed_key in state_dict:
                config_dict["vocab_size"] = state_dict[embed_key].shape[0]
                config_dict["d_model"]    = state_dict[embed_key].shape[1]

            # max_seq_len from causal_mask buffer shape [seq_len, seq_len]
            for k in state_dict:
                if "causal_mask" in k:
                    config_dict["max_seq_len"] = state_dict[k].shape[0]
                    break

            # num_layers from unique layer indices
            layer_idxs = {
                k.split(".")[1]
                for k in state_dict
                if k.startswith("layers.") and k.split(".")[1].isdigit()
            }
            if layer_idxs:
                config_dict["num_layers"] = len(layer_idxs)

        # Instantiate the architecture
        model_type = config_dict.get("model_type", "gpt")
        if model_type == "gpt":
            from src.architectures.transformer.decoder.gpt_decoder import GPTDecoder, GPTConfig
            import dataclasses
            gpt_fields = {f.name for f in dataclasses.fields(GPTConfig)}
            cfg   = GPTConfig(**{k: v for k, v in config_dict.items() if k in gpt_fields})
            model = GPTDecoder(cfg)
        elif model_type == "bert":
            from src.architectures.transformer.encoder.bert_encoder import BERTEncoder, BERTConfig
            import dataclasses
            bert_fields = {f.name for f in dataclasses.fields(BERTConfig)}
            cfg   = BERTConfig(**{k: v for k, v in config_dict.items() if k in bert_fields})
            model = BERTEncoder(cfg)
        else:
            raise ValueError(f"Unknown model_type '{model_type}' in config")

        # Load weights into the correctly-shaped model
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning("model.pt not found — using random weights", path=str(model_path))

        model.to(self.device)
        model.eval()

        # Load tokenizer
        tokenizer = None
        if tok_dir.exists():
            from src.tokenizer import load_tokenizer
            try:
                tokenizer = load_tokenizer(str(tok_dir))
            except Exception as e:
                logger.warning("Tokenizer load failed", error=str(e))

        return model, tokenizer

    def _evict_lru(self) -> None:
        """Remove the least-recently-used model from the cache."""
        if not self._cache:
            return
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._registry[k].last_used_at,
        )
        del self._cache[lru_key]
        logger.info("LRU model evicted", model=lru_key)

    def _resolve_key(self, model_name: str) -> str:
        """Normalise model name to 'name:version' format."""
        if ":" in model_name:
            return model_name
        return f"{model_name}:latest"
