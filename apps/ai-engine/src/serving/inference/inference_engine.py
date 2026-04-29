"""
InsightSerenity AI Engine — Inference Engine
=============================================
The inference engine is the core execution layer that runs model forward
passes for text generation, embeddings, and classification.

Responsibilities:
    - Device management: move tensors to the correct device
    - Batch assembly: combine multiple requests into one forward pass
    - Token generation: manage the autoregressive decoding loop
    - Embedding extraction: run encoder forward pass for dense vectors
    - Mixed precision: use bfloat16 for faster inference when supported

Key design decisions:
    - Stateless: no mutable state between requests (thread-safe)
    - Model-agnostic: works with GPTDecoder and BERTEncoder
    - Synchronous: async batching is handled by the layer above (dynamic_batcher)

Performance:
    The bottleneck is the transformer forward pass. On GPU with AMP,
    a single token generation step for a 125M model takes ~10ms.
    Dynamic batching (phase 10) groups concurrent requests to amortise this.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.config.settings import settings
from src.llm.inference.sampler import Sampler
from src.tokenizer.special_tokens import SpecialTokens as ST
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationRequest:
    """A single text generation request."""
    prompt:         str
    max_new_tokens: int   = 256
    temperature:    float = 1.0
    top_p:          float = 0.9
    top_k:          int   = 50
    strategy:       str   = "top_p"
    stop_sequences: List[str] = None
    stream:         bool  = False

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


@dataclass
class GenerationResult:
    """Result of a text generation call."""
    text:            str
    prompt_tokens:   int
    completion_tokens: int
    finish_reason:   str    # "stop" | "length" | "error"
    model:           str    = ""


@dataclass
class EmbeddingRequest:
    """A request for dense embeddings."""
    texts:       List[str]
    batch_size:  int = 32


@dataclass
class EmbeddingResult:
    """Result of an embedding call."""
    embeddings:          List[List[float]]
    model:               str = ""
    dim:                 int = 0
    total_prompt_tokens: int = 0   # Exact token count across all inputs


class InferenceEngine:
    """
    Runs model inference for text generation and embeddings.

    Args:
        model:      Loaded PyTorch model (GPTDecoder or BERTEncoder).
        tokenizer:  Loaded tokenizer.
        device:     Compute device. Defaults to settings device.
        use_amp:    Use automatic mixed precision for faster inference.
    """

    def __init__(
        self,
        model:     Any,
        tokenizer: Any,
        device:    Optional[str] = None,
        use_amp:   bool          = True,
    ) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = torch.device(device or str(settings.torch_device))
        self.use_amp   = use_amp and (self.device.type in ("cuda", "mps"))

        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "InferenceEngine ready",
            device=str(self.device),
            amp=self.use_amp,
        )

    # ── Text generation ────────────────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate text for a single request.

        Args:
            request: GenerationRequest with prompt and sampling parameters.

        Returns:
            GenerationResult with generated text and token counts.
        """
        sampler = Sampler(
            strategy=request.strategy,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        prompt_ids  = self._encode_prompt(request.prompt)
        input_ids   = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_len  = len(prompt_ids)
        generated   = []
        finish      = "length"

        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=settings.amp_dtype_torch)
            if self.use_amp
            else torch.no_grad()
        )

        with amp_ctx, torch.no_grad():
            for _ in range(request.max_new_tokens):
                output  = self.model(input_ids)
                logits  = output["logits"][0, -1, :]   # (V,) last position

                next_id = sampler.sample(logits)
                generated.append(next_id)

                if next_id == ST.EOS_ID or next_id == ST.END_TURN_ID:
                    finish = "stop"
                    break

                # Check stop sequences
                current_text = self.tokenizer.decode(generated, skip_special_tokens=True)
                if any(s in current_text for s in request.stop_sequences):
                    finish = "stop"
                    break

                # Append and continue
                next_t    = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                input_ids = torch.cat([input_ids, next_t], dim=1)

                # Safety: don't exceed model's context length
                if input_ids.shape[1] >= getattr(self.model, "config", type("c", (), {"max_seq_len": 2048})()).max_seq_len:
                    finish = "length"
                    break

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=len(generated),
            finish_reason=finish,
        )

    def stream_generate(
        self, request: GenerationRequest
    ) -> Iterator[str]:
        """
        Generate text token-by-token, yielding each decoded token.

        Used for server-sent events (SSE) streaming responses.

        Args:
            request: GenerationRequest.

        Yields:
            Decoded token strings one at a time.
        """
        sampler = Sampler(
            strategy=request.strategy,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        prompt_ids = self._encode_prompt(request.prompt)
        input_ids  = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated  = []

        with torch.no_grad():
            for _ in range(request.max_new_tokens):
                output  = self.model(input_ids)
                logits  = output["logits"][0, -1, :]

                all_ids = torch.tensor(
                    prompt_ids + generated, dtype=torch.long, device=self.device
                )
                next_id = sampler.sample(logits, all_ids)
                generated.append(next_id)

                if next_id in (ST.EOS_ID, ST.END_TURN_ID):
                    break

                token_str = self.tokenizer.decode([next_id], skip_special_tokens=True)

                current_text = self.tokenizer.decode(generated, skip_special_tokens=True)
                if any(s in current_text for s in request.stop_sequences):
                    break

                if token_str:
                    yield token_str

                next_t    = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                input_ids = torch.cat([input_ids, next_t], dim=1)

                if input_ids.shape[1] >= 2048:
                    break

    # ── Embeddings ─────────────────────────────────────────────────────────────

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """
        Compute dense embeddings for a list of texts.

        Uses the model's encoder hidden states — specifically the [CLS] token
        for BERT-style models, or the last token for GPT-style models.

        Args:
            request: EmbeddingRequest with texts and batch size.

        Returns:
            EmbeddingResult with embeddings as float lists.
        """
        all_embeddings:    List[List[float]] = []
        total_prompt_tokens = 0

        for i in range(0, len(request.texts), request.batch_size):
            batch_texts  = request.texts[i:i + request.batch_size]
            batch_embs, batch_tokens = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embs)
            total_prompt_tokens += batch_tokens

        dim = len(all_embeddings[0]) if all_embeddings else 0
        return EmbeddingResult(
            embeddings=all_embeddings,
            dim=dim,
            total_prompt_tokens=total_prompt_tokens,
        )

    def _embed_batch(self, texts: List[str]) -> tuple:
        """Embed a small batch of texts. Returns (embeddings, total_tokens)."""
        # Encode and pad to same length
        encoded   = [self.tokenizer.encode(t, add_special_tokens=True) for t in texts]
        total_tok = sum(len(e) for e in encoded)
        max_len   = max(len(e) for e in encoded)
        input_ids = torch.zeros(len(encoded), max_len, dtype=torch.long, device=self.device)
        attn_mask = torch.zeros(len(encoded), max_len, dtype=torch.long, device=self.device)

        for i, enc in enumerate(encoded):
            input_ids[i, :len(enc)] = torch.tensor(enc, dtype=torch.long)
            attn_mask[i, :len(enc)] = 1

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attn_mask)

        # Extract embeddings: use pooler_output (BERT) or mean of last_hidden
        if "pooler_output" in output:
            embs = output["pooler_output"]   # (B, D)
        else:
            hidden   = output["last_hidden"]                     # (B, T, D)
            mask_exp = attn_mask.unsqueeze(-1).float()
            embs     = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1)  # mean pool

        # L2 normalise for cosine similarity downstream
        embs = F.normalize(embs, dim=-1)
        return embs.cpu().float().tolist(), total_tok

    # ── Internal ───────────────────────────────────────────────────────────────

    def _encode_prompt(self, prompt: str) -> List[int]:
        """Encode prompt text to token IDs."""
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer loaded")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return ids
