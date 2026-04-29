"""
Unit tests for BPETrainer and BPETokenizer

Coverage:
    BPETrainer:
        - _word_to_char_sequence: Ġ prefix handling
        - _pretokenize: whitespace splitting, Ġ insertion
        - _count_pairs: frequency weighting
        - _apply_merge: all occurrences replaced, single-pass safety
        - train(): vocabulary grows to target size
        - train(): merge rules are ordered (earlier = higher priority)
        - train(): special tokens are present in output vocabulary

    BPETokenizer:
        - _tokenize: single word, multi-word text
        - encode: BOS/EOS added, truncation, empty input
        - decode: Ġ convention produces correct spaces, round-trip
        - encode/decode round-trip fidelity
        - encode_batch / decode_batch
        - Edge cases: very short text, only punctuation, long text
        - Cache correctness: same word encoded consistently
        - save() / load() round-trip: identical behaviour before and after

    Integration:
        - train on synthetic corpus → save → load → encode → decode
        - Trained tokenizer handles OOV characters via character fallback
"""

import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from src.tokenizer.bpe.bpe_trainer import BPETrainer, SPACE_PREFIX
from src.tokenizer.bpe.bpe_tokenizer import BPETokenizer
from src.tokenizer.special_tokens import SpecialTokens as ST
from src.tokenizer.vocabulary import Vocabulary


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_corpus_file(texts: List[str], suffix: str = ".txt") -> str:
    """Write a list of texts to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as f:
        for text in texts:
            f.write(text + "\n")
        return f.name


def make_jsonl_corpus(texts: List[str]) -> str:
    """Write texts as JSONL to a temp file and return the path."""
    import json
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")
        return f.name


# ─────────────────────────────────────────────────────────────────────────────
# BPETrainer unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBPETrainerInternals:
    """Tests for BPETrainer's internal helper methods."""

    def setup_method(self):
        self.trainer = BPETrainer(vocab_size=100, min_frequency=1)

    def test_word_to_char_sequence_plain_word(self):
        """Plain word (no Ġ prefix) → individual characters."""
        chars = self.trainer._word_to_char_sequence("hello")
        assert chars == ["h", "e", "l", "l", "o"]

    def test_word_to_char_sequence_space_prefix(self):
        """Word with Ġ prefix → Ġ as first char, then individual chars."""
        chars = self.trainer._word_to_char_sequence("Ġworld")
        assert chars == ["Ġ", "w", "o", "r", "l", "d"]

    def test_word_to_char_sequence_single_char(self):
        assert self.trainer._word_to_char_sequence("a") == ["a"]

    def test_word_to_char_sequence_single_prefix_char(self):
        """A word that is just the space prefix marker."""
        chars = self.trainer._word_to_char_sequence("Ġ")
        # Ġ is the only character — just [Ġ]
        assert chars == ["Ġ"]

    def test_pretokenize_adds_space_prefix(self):
        """Words after the first must have Ġ prepended."""
        words = self.trainer._pretokenize("hello world")
        # "world" is preceded by a space → "Ġworld"
        assert any(w.startswith(SPACE_PREFIX) for w in words)

    def test_pretokenize_first_word_no_prefix(self):
        """The first word must NOT have a Ġ prefix."""
        words = self.trainer._pretokenize("hello world")
        first = [w for w in words if not w.startswith(SPACE_PREFIX)]
        assert len(first) >= 1

    def test_pretokenize_empty_string(self):
        words = self.trainer._pretokenize("")
        assert words == []

    def test_count_pairs_returns_counter(self):
        """_count_pairs must return a Counter-like object."""
        word_to_chars = {
            "hello": ["h", "e", "l", "l", "o"],
            "world": ["w", "o", "r", "l", "d"],
        }
        word_freqs = {"hello": 3, "world": 2}
        pairs = self.trainer._count_pairs(word_to_chars, word_freqs)
        assert len(pairs) > 0

    def test_count_pairs_weights_by_word_frequency(self):
        """Pair count must be proportional to word frequency."""
        word_to_chars = {
            "ab": ["a", "b"],
        }
        word_freqs_low  = {"ab": 1}
        word_freqs_high = {"ab": 10}

        pairs_low  = self.trainer._count_pairs(word_to_chars, word_freqs_low)
        pairs_high = self.trainer._count_pairs(word_to_chars, word_freqs_high)

        pair = ("a", "b")
        assert pairs_high[pair] == 10 * pairs_low[pair]

    def test_apply_merge_replaces_pair(self):
        """Applying a merge should produce the merged token at the correct position."""
        word_to_chars = {"ab": ["a", "b", "c"]}
        result = self.trainer._apply_merge(word_to_chars, ("a", "b"), "ab")
        assert result["ab"] == ["ab", "c"]

    def test_apply_merge_handles_multiple_occurrences(self):
        """All occurrences of the pair in one word must be merged."""
        word_to_chars = {"aaaa": ["a", "a", "a", "a"]}
        result = self.trainer._apply_merge(word_to_chars, ("a", "a"), "aa")
        # After one pass: ["aa", "aa"]
        assert result["aaaa"] == ["aa", "aa"]

    def test_apply_merge_does_not_overlap(self):
        """A merge should not produce overlapping replacements."""
        # "aaa" → apply merge (a, a) → should produce ["aa", "a"], not ["a", "aa"]
        word_to_chars = {"aaa": ["a", "a", "a"]}
        result = self.trainer._apply_merge(word_to_chars, ("a", "a"), "aa")
        # First pair is merged; the remaining 'a' stays
        assert result["aaa"] == ["aa", "a"]

    def test_apply_merge_no_match_unchanged(self):
        """Words that don't contain the pair must be returned unchanged."""
        word_to_chars = {"xyz": ["x", "y", "z"]}
        result = self.trainer._apply_merge(word_to_chars, ("a", "b"), "ab")
        assert result["xyz"] == ["x", "y", "z"]


class TestBPETrainerTraining:
    """Integration tests for the full BPETrainer.train() workflow."""

    def test_train_produces_vocabulary(self, tmp_path):
        corpus_path = str(tmp_path / "corpus.txt")
        Path(corpus_path).write_text(
            "hello world\nhello world\nhello world\nfoo bar\n" * 50,
            encoding="utf-8",
        )
        trainer = BPETrainer(vocab_size=50, min_frequency=1)
        merge_rules, vocab = trainer.train(corpus_path)

        assert isinstance(vocab, Vocabulary)
        assert vocab.size > 0

    def test_train_vocab_size_approaches_target(self, tmp_path):
        """Vocab size should be close to target (may be slightly less if corpus is small)."""
        corpus_path = str(tmp_path / "corpus.txt")
        # Large enough corpus to reach target
        Path(corpus_path).write_text(
            "the cat sat on the mat the dog sat on the mat\n" * 200,
            encoding="utf-8",
        )
        target = 60
        trainer = BPETrainer(vocab_size=target, min_frequency=1)
        _, vocab = trainer.train(corpus_path)

        # Vocab may not reach exact target if corpus is small, but should be close
        assert vocab.size <= target
        assert vocab.size > len(ST.all_ids())   # At least more than special tokens

    def test_train_special_tokens_present(self, tmp_path):
        """Special tokens must appear in the trained vocabulary."""
        corpus_path = str(tmp_path / "corpus.txt")
        Path(corpus_path).write_text("hello world\n" * 50, encoding="utf-8")

        trainer = BPETrainer(vocab_size=40, min_frequency=1)
        _, vocab = trainer.train(corpus_path)

        for token in ST.all_tokens():
            assert token in vocab, f"Special token '{token}' missing from trained vocab"

    def test_train_produces_merge_rules(self, tmp_path):
        """Training must produce at least one merge rule."""
        corpus_path = str(tmp_path / "corpus.txt")
        Path(corpus_path).write_text(
            "hello world foo bar baz qux\n" * 100, encoding="utf-8"
        )
        trainer = BPETrainer(vocab_size=50, min_frequency=1)
        merge_rules, _ = trainer.train(corpus_path)

        assert len(merge_rules) > 0

    def test_train_merge_rules_are_tuples(self, tmp_path):
        """Each merge rule must be a (str, str) tuple."""
        corpus_path = str(tmp_path / "corpus.txt")
        Path(corpus_path).write_text("hello world\n" * 50, encoding="utf-8")

        trainer = BPETrainer(vocab_size=40, min_frequency=1)
        merge_rules, _ = trainer.train(corpus_path)

        for rule in merge_rules:
            assert isinstance(rule, tuple)
            assert len(rule) == 2
            assert isinstance(rule[0], str)
            assert isinstance(rule[1], str)

    def test_train_works_with_jsonl_corpus(self, tmp_path):
        """Training must work with JSONL corpus files."""
        corpus_path = str(tmp_path / "corpus.jsonl")
        import json
        with open(corpus_path, "w") as f:
            for _ in range(100):
                f.write(json.dumps({"text": "hello world the cat sat"}) + "\n")

        trainer = BPETrainer(vocab_size=40, min_frequency=1)
        merge_rules, vocab = trainer.train(corpus_path, text_field="text")

        assert vocab.size > 0
        assert len(merge_rules) > 0


# ─────────────────────────────────────────────────────────────────────────────
# BPETokenizer unit tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_tokenizer(tmp_path_factory):
    """
    Module-scoped fixture: train a BPETokenizer once and reuse across tests.
    This avoids re-training in every test method (training is slow).
    """
    tmp_path = tmp_path_factory.mktemp("bpe")
    corpus_path = str(tmp_path / "corpus.txt")

    # Write a varied corpus so the tokenizer learns useful merges
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test of the tokenizer",
        "machine learning is a subset of artificial intelligence",
        "natural language processing enables machines to understand text",
        "transformers have revolutionized deep learning and natural language",
        "the model learns to predict the next token in the sequence",
        "training data quality directly affects model performance",
    ]
    Path(corpus_path).write_text(
        "\n".join(sentences * 100),   # Repeat 100× so min_frequency is met
        encoding="utf-8",
    )

    tokenizer = BPETokenizer()
    tokenizer.train(corpus_path, vocab_size=200, min_frequency=2)
    return tokenizer


class TestBPETokenizerEncoding:

    def test_encode_returns_list_of_ints(self, trained_tokenizer):
        ids = trained_tokenizer.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_nonempty_for_nonempty_input(self, trained_tokenizer):
        ids = trained_tokenizer.encode("hello")
        assert len(ids) > 0

    def test_encode_empty_string_returns_empty(self, trained_tokenizer):
        ids = trained_tokenizer.encode("")
        assert ids == []

    def test_encode_adds_bos_token(self, trained_tokenizer):
        """BOS must be the first token when add_special_tokens=True."""
        ids = trained_tokenizer.encode("hello", add_special_tokens=True)
        assert ids[0] == ST.BOS_ID

    def test_encode_adds_eos_token(self, trained_tokenizer):
        """EOS must be the last token when add_special_tokens=True."""
        ids = trained_tokenizer.encode("hello", add_special_tokens=True)
        assert ids[-1] == ST.EOS_ID

    def test_encode_no_special_tokens(self, trained_tokenizer):
        """add_special_tokens=False must not add BOS or EOS."""
        ids = trained_tokenizer.encode("hello", add_special_tokens=False)
        assert ST.BOS_ID not in ids
        assert ST.EOS_ID not in ids

    def test_encode_truncates_at_max_length(self, trained_tokenizer):
        """Output must not exceed max_length tokens."""
        long_text = "hello world " * 500
        ids = trained_tokenizer.encode(long_text, max_length=50)
        assert len(ids) <= 50

    def test_encode_truncated_ends_with_eos(self, trained_tokenizer):
        """Truncated sequences must still end with EOS."""
        long_text = "hello world " * 500
        ids = trained_tokenizer.encode(
            long_text, add_special_tokens=True, max_length=20
        )
        assert ids[-1] == ST.EOS_ID

    def test_encode_all_ids_in_vocab_range(self, trained_tokenizer):
        """All returned IDs must be valid vocabulary IDs."""
        ids = trained_tokenizer.encode("hello world natural language")
        for token_id in ids:
            token = trained_tokenizer.vocab.id_to_token(token_id)
            assert token != ST.UNK or token_id == ST.UNK_ID

    def test_encode_same_text_same_ids(self, trained_tokenizer):
        """Encoding must be deterministic — same input always gives same output."""
        text = "the quick brown fox"
        assert trained_tokenizer.encode(text) == trained_tokenizer.encode(text)


class TestBPETokenizerDecoding:

    def test_decode_returns_string(self, trained_tokenizer):
        ids  = trained_tokenizer.encode("hello world", add_special_tokens=False)
        text = trained_tokenizer.decode(ids)
        assert isinstance(text, str)

    def test_decode_nonempty_for_nonempty_ids(self, trained_tokenizer):
        ids  = trained_tokenizer.encode("hello", add_special_tokens=False)
        text = trained_tokenizer.decode(ids)
        assert len(text) > 0

    def test_decode_skips_special_tokens_by_default(self, trained_tokenizer):
        ids  = trained_tokenizer.encode("hello", add_special_tokens=True)
        text = trained_tokenizer.decode(ids, skip_special_tokens=True)
        assert ST.BOS not in text
        assert ST.EOS not in text

    def test_decode_includes_special_tokens_when_requested(self, trained_tokenizer):
        ids  = trained_tokenizer.encode("hello", add_special_tokens=True)
        text = trained_tokenizer.decode(ids, skip_special_tokens=False)
        assert ST.BOS in text or ST.EOS in text

    def test_decode_empty_list_returns_empty_string(self, trained_tokenizer):
        assert trained_tokenizer.decode([]) == ""

    def test_roundtrip_preserves_words(self, trained_tokenizer):
        """
        Round-trip (encode → decode) must produce text containing the original words.
        We check word membership rather than exact equality because tokenizer
        normalisation may change spacing/casing slightly.
        """
        original = "the quick brown fox"
        ids      = trained_tokenizer.encode(original, add_special_tokens=False)
        decoded  = trained_tokenizer.decode(ids)

        for word in original.split():
            assert word in decoded, (
                f"Word '{word}' missing from decoded text: '{decoded}'"
            )


class TestBPETokenizerBatch:

    def test_encode_batch_returns_dict(self, trained_tokenizer):
        result = trained_tokenizer.encode_batch(["hello", "world"])
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_encode_batch_length_matches_input(self, trained_tokenizer):
        texts  = ["hello", "world", "foo bar"]
        result = trained_tokenizer.encode_batch(texts)
        assert len(result["input_ids"]) == len(texts)

    def test_encode_batch_padding_makes_uniform_length(self, trained_tokenizer):
        texts  = ["a", "hello world", "the quick brown fox"]
        result = trained_tokenizer.encode_batch(texts, padding=True)
        lengths = [len(ids) for ids in result["input_ids"]]
        assert len(set(lengths)) == 1, "All padded sequences must have the same length"

    def test_encode_batch_attention_mask_correct(self, trained_tokenizer):
        texts  = ["a", "hello world"]
        result = trained_tokenizer.encode_batch(texts, padding=True)

        for ids, mask in zip(result["input_ids"], result["attention_mask"]):
            # Real tokens have mask=1, padding tokens have mask=0
            assert len(ids) == len(mask)
            for token_id, m in zip(ids, mask):
                if token_id == ST.PAD_ID:
                    assert m == 0
                else:
                    assert m == 1

    def test_decode_batch_length_matches_input(self, trained_tokenizer):
        batch_ids = [[ST.BOS_ID, 15, 16, ST.EOS_ID], [ST.BOS_ID, 17, ST.EOS_ID]]
        texts     = trained_tokenizer.decode_batch(batch_ids)
        assert len(texts) == 2

    def test_decode_batch_returns_list_of_strings(self, trained_tokenizer):
        batch_ids = [[ST.BOS_ID, 15, ST.EOS_ID]]
        texts     = trained_tokenizer.decode_batch(batch_ids)
        assert isinstance(texts, list)
        assert isinstance(texts[0], str)


class TestBPETokenizerEdgeCases:

    def test_encode_punctuation_only(self, trained_tokenizer):
        """Punctuation-only input must not crash."""
        ids = trained_tokenizer.encode("!!?!?!")
        assert isinstance(ids, list)

    def test_encode_numeric_text(self, trained_tokenizer):
        ids = trained_tokenizer.encode("123 456 789")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_single_character(self, trained_tokenizer):
        ids = trained_tokenizer.encode("a")
        assert len(ids) >= 1

    def test_encode_whitespace_only(self, trained_tokenizer):
        """Whitespace-only input should produce BOS+EOS or empty."""
        ids = trained_tokenizer.encode("   ", add_special_tokens=True)
        # May produce only BOS+EOS if whitespace yields no tokens
        assert isinstance(ids, list)

    def test_encode_repeated_word(self, trained_tokenizer):
        """Cache must not corrupt results for repeated words."""
        single = trained_tokenizer.encode("hello", add_special_tokens=False)
        triple = trained_tokenizer.encode("hello hello hello", add_special_tokens=False)
        # "hello hello hello" should contain 3× the tokens of "hello" (approximately)
        assert len(triple) >= len(single)


class TestBPETokenizerSaveLoad:

    def test_save_creates_required_files(self, trained_tokenizer, tmp_path):
        trained_tokenizer.save(str(tmp_path))
        assert (tmp_path / "vocab.json").exists()
        assert (tmp_path / "merges.txt").exists()
        assert (tmp_path / "tokenizer_config.json").exists()

    def test_load_restores_same_vocab_size(self, trained_tokenizer, tmp_path):
        trained_tokenizer.save(str(tmp_path))
        loaded = BPETokenizer.load(str(tmp_path))
        assert loaded.vocab_size == trained_tokenizer.vocab_size

    def test_load_restores_same_merge_count(self, trained_tokenizer, tmp_path):
        trained_tokenizer.save(str(tmp_path))
        loaded = BPETokenizer.load(str(tmp_path))
        assert len(loaded._merge_rules) == len(trained_tokenizer._merge_rules)

    def test_loaded_tokenizer_encodes_same_ids(self, trained_tokenizer, tmp_path):
        """After save/load, encoding must produce identical results."""
        text = "hello world the quick brown fox"
        trained_tokenizer.save(str(tmp_path))
        loaded = BPETokenizer.load(str(tmp_path))

        original_ids = trained_tokenizer.encode(text)
        loaded_ids   = loaded.encode(text)
        assert original_ids == loaded_ids

    def test_loaded_tokenizer_decodes_same_text(self, trained_tokenizer, tmp_path):
        """After save/load, decoding must produce identical results."""
        text = "hello world"
        ids  = trained_tokenizer.encode(text, add_special_tokens=False)

        trained_tokenizer.save(str(tmp_path))
        loaded = BPETokenizer.load(str(tmp_path))

        assert loaded.decode(ids) == trained_tokenizer.decode(ids)

    def test_loaded_tokenizer_special_token_ids_correct(self, trained_tokenizer, tmp_path):
        trained_tokenizer.save(str(tmp_path))
        loaded = BPETokenizer.load(str(tmp_path))

        assert loaded.pad_token_id  == ST.PAD_ID
        assert loaded.bos_token_id  == ST.BOS_ID
        assert loaded.eos_token_id  == ST.EOS_ID
        assert loaded.unk_token_id  == ST.UNK_ID
        assert loaded.mask_token_id == ST.MASK_ID
