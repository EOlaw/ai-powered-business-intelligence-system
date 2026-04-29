"""
InsightSerenity AI Engine — SentencePiece Wrapper
==================================================
Wraps Google's SentencePiece library behind the BaseTokenizer interface so
SentencePiece models can be used anywhere a BaseTokenizer is expected —
training loops, inference servers, the SDK — with zero code changes.

Why SentencePiece as a third option?
    - Language-agnostic: handles CJK, Arabic, Indic scripts without
      language-specific pre-tokenisation rules
    - Byte-fallback: SentencePiece's BPE variant (SentencePiece BPE) can
      represent ANY unicode character via byte-level fallback, guaranteeing
      zero unknown tokens
    - Used by LLaMA, T5, mT5, Gemma, and many multilingual models
    - Training is fast (C++ implementation)

Important: this wrapper requires `sentencepiece` to be installed separately
because it is a C extension with platform-specific builds:
    pip install sentencepiece

If sentencepiece is not installed and you try to use this class, a clear
ImportError with installation instructions is raised.

Architecture note:
    SentencePiece manages its own vocabulary internally (in a .model file).
    We bridge it to our Vocabulary class by syncing token strings and IDs.
    Special tokens are inserted at the positions matching SpecialTokens.

Files saved by save():
    spiece.model          — binary SentencePiece model (opaque, owned by sp)
    vocab.json            — our Vocabulary (synced from the sp model)
    tokenizer_config.json — class name, config, and sp model path

Usage:
    from src.tokenizer.sentencepiece.sp_wrapper import SentencePieceWrapper

    tokenizer = SentencePieceWrapper()
    tokenizer.train("storage/datasets/corpus.jsonl", vocab_size=32_000)
    tokenizer.save("storage/tokenizers/spiece-32k/")

    tokenizer = SentencePieceWrapper.load("storage/tokenizers/spiece-32k/")
    ids  = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(ids)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from src.tokenizer.base.base_tokenizer import BaseTokenizer
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.special_tokens import SpecialTokens as ST, ALL_SPECIAL_TOKENS
from src.utils.file_io import iter_jsonl, write_json, read_json, ensure_dir, write_text
from src.utils.logger import get_logger, LogTimer

logger = get_logger(__name__)

# Lazy import guard — gives a clear error if sentencepiece is not installed
_SP_AVAILABLE = False
try:
    import sentencepiece as spm
    _SP_AVAILABLE = True
except ImportError:
    spm = None  # type: ignore[assignment]


def _require_sentencepiece() -> None:
    """Raise a descriptive ImportError if sentencepiece is not installed."""
    if not _SP_AVAILABLE:
        raise ImportError(
            "The sentencepiece library is required to use SentencePieceWrapper.\n"
            "Install it with:  pip install sentencepiece\n"
            "SentencePiece is a C extension and must be installed for your platform."
        )


class SentencePieceWrapper(BaseTokenizer):
    """
    Wraps a SentencePiece model behind the BaseTokenizer interface.

    Supports both SentencePiece BPE and Unigram model types. The default
    is BPE with byte-fallback, which is the same configuration used by
    LLaMA-2/3, Mistral, and Gemma.

    Args:
        model_path:    Path to a pre-trained .model file. If None, the
                       tokenizer must be trained before use.
        model_type:    "bpe" or "unigram". BPE is generally recommended
                       for monolingual English; Unigram for multilingual.
        byte_fallback: Enable byte-level fallback so no token is ever <unk>.
                       Highly recommended. Default True.
        add_bos_token: Prepend <bos> on encode. Default True.
        add_eos_token: Append <eos> on encode. Default True.
        max_length:    Maximum token sequence length.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_type: str = "bpe",
        byte_fallback: bool = True,
        add_bos_token: bool = True,
        add_eos_token: bool = True,
        max_length: int = 2048,
    ) -> None:
        super().__init__(
            vocab=None,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            max_length=max_length,
        )
        self._model_type    = model_type
        self._byte_fallback = byte_fallback
        self._sp_model      = None   # type: ignore[assignment]  # loaded by _load_model or train

        if model_path:
            _require_sentencepiece()
            self._load_model(str(model_path))

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        corpus_path: str,
        vocab_size: int = 32_000,
        text_field: str = "text",
        character_coverage: float = 0.9995,
        num_threads: int = 4,
        **kwargs,
    ) -> None:
        """
        Train a SentencePiece model on a text corpus.

        Writes a temporary plain-text file from the JSONL corpus, trains
        the SentencePiece model, then cleans up the temp file.

        Args:
            corpus_path:         Path to JSONL or plain text corpus.
            vocab_size:          Target vocabulary size.
            text_field:          For JSONL: key containing the document text.
            character_coverage:  Fraction of characters covered by the model.
                                 0.9995 is good for Latin scripts; use 0.9999
                                 for languages with large character sets.
            num_threads:         Parallel training threads.
        """
        _require_sentencepiece()

        import tempfile
        import os

        logger.info(
            "SentencePiece training started",
            corpus=corpus_path,
            vocab_size=vocab_size,
            model_type=self._model_type,
        )

        # SentencePiece requires plain text input — extract from JSONL if needed
        if corpus_path.endswith(".jsonl") or corpus_path.endswith(".jsonl.gz"):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp_path = tmp.name
                for record in iter_jsonl(corpus_path):
                    text = record.get(text_field, "")
                    if text:
                        tmp.write(text.replace("\n", " ") + "\n")
            input_path = tmp_path
        else:
            input_path = corpus_path
            tmp_path   = None

        # Build the special tokens list to pass to SentencePiece
        # (we control IDs 0–11 ourselves)
        user_defined_symbols = [
            st.token for st in ALL_SPECIAL_TOKENS
            if st.token not in ("<pad>", "<unk>")   # sp handles pad/unk natively
        ]

        # Train model to a temp directory, then load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_prefix = os.path.join(tmpdir, "spiece")

            train_args = {
                "input":                   input_path,
                "model_prefix":            model_prefix,
                "vocab_size":              vocab_size,
                "model_type":              self._model_type,
                "character_coverage":      character_coverage,
                "num_threads":             num_threads,
                "pad_id":                  ST.PAD_ID,
                "unk_id":                  ST.UNK_ID,
                "bos_id":                  ST.BOS_ID,
                "eos_id":                  ST.EOS_ID,
                "pad_piece":               ST.PAD,
                "unk_piece":               ST.UNK,
                "bos_piece":               ST.BOS,
                "eos_piece":               ST.EOS,
                "user_defined_symbols":    ",".join(user_defined_symbols),
                "byte_fallback":           self._byte_fallback,
                "unk_surface":             " ⁇ ",   # Visual marker for unknowns
                "normalization_rule_name": "nmt_nfkc_cf",  # Unicode normalization
            }

            with LogTimer(logger, "SentencePiece model training"):
                spm.SentencePieceTrainer.train(**train_args)   # type: ignore[union-attr]

            # Load the trained model
            self._load_model(model_prefix + ".model")

        # Clean up the temporary plain-text corpus file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        logger.info(
            "SentencePiece training complete",
            vocab_size=self.vocab_size,
            model_type=self._model_type,
        )

    # ── Tokenisation ───────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using the SentencePiece model.

        Delegates directly to the underlying sp model's encode_as_pieces()
        method, which handles pre-tokenisation, BPE/Unigram segmentation,
        and special character normalisation internally.

        Args:
            text: Input text string.

        Returns:
            List of sub-word piece strings.
        """
        if self._sp_model is None:
            raise RuntimeError(
                "SentencePieceWrapper is not trained. Call train() or "
                "load a model via SentencePieceWrapper.load(directory)."
            )
        # Encode to piece strings (e.g. ["▁Hello", ",", "▁world", "!"])
        # SentencePiece uses ▁ (U+2581) as its word-start marker (vs our Ġ)
        return self._sp_model.encode_as_pieces(text)   # type: ignore[union-attr]

    def _tokens_to_string(self, tokens: List[str]) -> str:
        """
        Decode SentencePiece tokens back to text.

        SentencePiece's ▁ prefix encodes spaces. Removing ▁ and joining
        with appropriate spaces reconstructs the original text.
        """
        if self._sp_model is None:
            # Fallback: naive string join
            return "".join(t.replace("▁", " ") for t in tokens).strip()

        return self._sp_model.decode_pieces(tokens)   # type: ignore[union-attr]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: Union[str, Path]) -> None:
        """
        Save the SentencePiece model and metadata to `directory`.

        Files created:
            spiece.model          — binary SentencePiece model
            vocab.json            — our Vocabulary (synced from sp model)
            tokenizer_config.json — class name and config
        """
        _require_sentencepiece()

        if self._sp_model is None:
            raise RuntimeError("No model to save. Train a model first.")

        directory = ensure_dir(directory)

        # Write the binary SentencePiece model
        model_path = directory / "spiece.model"
        with open(model_path, "wb") as f:
            f.write(self._sp_model.serialized_model_proto())   # type: ignore[union-attr]

        # Sync vocabulary from the sp model
        self._sync_vocabulary()
        self._vocab.save(directory / "vocab.json")

        # Write config
        write_json(directory / "tokenizer_config.json", {
            "tokenizer_class":    "SentencePieceWrapper",
            "model_type":         self._model_type,
            "vocab_size":         self.vocab_size,
            "byte_fallback":      self._byte_fallback,
            "add_bos_token":      self._add_bos,
            "add_eos_token":      self._add_eos,
            "max_length":         self._max_length,
            "sp_model_file":      "spiece.model",
        })

        logger.info(
            "SentencePieceWrapper saved",
            directory=str(directory),
            vocab_size=self.vocab_size,
        )

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "SentencePieceWrapper":
        """
        Load a SentencePieceWrapper from a directory previously saved by save().

        Args:
            directory: Directory containing spiece.model and tokenizer_config.json.

        Returns:
            Initialised SentencePieceWrapper ready for use.
        """
        _require_sentencepiece()

        directory = Path(directory)
        config    = read_json(directory / "tokenizer_config.json")

        tokenizer = cls(
            model_path=directory / config.get("sp_model_file", "spiece.model"),
            model_type=config.get("model_type", "bpe"),
            byte_fallback=config.get("byte_fallback", True),
            add_bos_token=config.get("add_bos_token", True),
            add_eos_token=config.get("add_eos_token", True),
            max_length=config.get("max_length", 2048),
        )

        logger.info(
            "SentencePieceWrapper loaded",
            directory=str(directory),
            vocab_size=tokenizer.vocab_size,
        )
        return tokenizer

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_model(self, model_path: str) -> None:
        """
        Load a SentencePiece binary model file and sync the vocabulary.

        Args:
            model_path: Path to the .model file.
        """
        _require_sentencepiece()
        sp = spm.SentencePieceProcessor()   # type: ignore[union-attr]
        sp.Load(str(model_path))
        self._sp_model = sp
        self._sync_vocabulary()
        logger.debug(
            "SentencePiece model loaded",
            path=str(model_path),
            vocab_size=sp.GetPieceSize(),
        )

    def _sync_vocabulary(self) -> None:
        """
        Build our Vocabulary object by iterating over the SentencePiece
        model's piece table.

        This bridges SentencePiece's internal vocab representation to our
        Vocabulary class so the rest of the platform can use a unified API.
        """
        if self._sp_model is None:
            return

        self._vocab = Vocabulary()
        sp_vocab_size = self._sp_model.GetPieceSize()   # type: ignore[union-attr]

        for piece_id in range(sp_vocab_size):
            piece = self._sp_model.IdToPiece(piece_id)   # type: ignore[union-attr]
            # The SentencePiece vocab uses its own IDs; we add tokens in order
            # and the Vocabulary assigns IDs sequentially after special tokens
            if piece not in self._vocab:
                self._vocab.add_token(piece)
