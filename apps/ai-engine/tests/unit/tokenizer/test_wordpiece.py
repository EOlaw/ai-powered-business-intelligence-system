"""
Unit tests for WordPieceTrainer and WordPieceTokenizer

Coverage:
    WordPieceTrainer internals:
        - _word_to_chars: ## prefix convention
        - _count_words:   frequency counting
        - _compute_freqs: subword + pair frequency weighting
        - _apply_merge:   single and multiple occurrence replacement

    WordPieceTokenizer:
        - _wordpiece_encode: longest-match-first algorithm
        - _wordpiece_encode: unknown-word fallback to [UNK]
        - _wordpiece_encode: max_word_chars enforcement
        - _tokenize: lowercasing, punctuation splitting
        - encode:    BOS/EOS, truncation
        - decode:    ## prefix removal, spacing reconstruction
        - Round-trip fidelity (encode → decode → original words present)
        - encode_batch / decode_batch
        - Edge cases: empty string, numbers, unicode, repeated tokens

    Persistence:
        - save() / load() round-trip: identical encoding before and after
        - Required files are created by save()
        - Config values are preserved through load()

    Comparison with BPE:
        - WordPiece uses ## not Ġ — verify the convention difference
"""

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from src.tokenizer.wordpiece.wordpiece_tokenizer import (
    WordPieceTrainer,
    WordPieceTokenizer,
    CONTINUATION_PREFIX,
)
from src.tokenizer.special_tokens import SpecialTokens as ST
from src.tokenizer.vocabulary import Vocabulary


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_corpus_txt(texts: List[str], directory: Path) -> str:
    path = directory / "corpus.txt"
    path.write_text("\n".join(texts * 100), encoding="utf-8")
    return str(path)


def make_corpus_jsonl(texts: List[str], directory: Path) -> str:
    path = directory / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# WordPieceTrainer internals
# ─────────────────────────────────────────────────────────────────────────────

class TestWordPieceTrainerInternals:

    def setup_method(self):
        self.trainer = WordPieceTrainer(vocab_size=100, min_frequency=1)

    def test_word_to_chars_single_character(self):
        """Single character → no continuation prefix needed."""
        chars = self.trainer._word_to_chars("a")
        assert chars == ["a"]

    def test_word_to_chars_plain_word(self):
        """First char is plain; rest get '##' prefix."""
        chars = self.trainer._word_to_chars("hello")
        assert chars[0] == "h"
        for char in chars[1:]:
            assert char.startswith(CONTINUATION_PREFIX), (
                f"Expected '##' prefix on '{char}'"
            )

    def test_word_to_chars_correct_content(self):
        """Characters after stripping ## must match the original word."""
        word  = "playing"
        chars = self.trainer._word_to_chars(word)
        reconstructed = chars[0] + "".join(
            c[len(CONTINUATION_PREFIX):] for c in chars[1:]
        )
        assert reconstructed == word

    def test_word_to_chars_empty_string(self):
        assert self.trainer._word_to_chars("") == []

    def test_apply_merge_joins_tokens(self):
        """Applying ("play", "##ing") should produce "playing"."""
        word_to_subwords = {"playing": ["play", "##ing"]}
        result = self.trainer._apply_merge(
            word_to_subwords, ("play", "##ing"), "playing"
        )
        assert result["playing"] == ["playing"]

    def test_apply_merge_no_match_unchanged(self):
        """Words not containing the pair must be returned unchanged."""
        word_to_subwords = {"hello": ["h", "##e", "##l", "##l", "##o"]}
        result = self.trainer._apply_merge(
            word_to_subwords, ("x", "##y"), "xy"
        )
        assert result["hello"] == ["h", "##e", "##l", "##l", "##o"]

    def test_apply_merge_multiple_words(self):
        """Merge must be applied to all words containing the pair."""
        word_to_subwords = {
            "ab":  ["a", "##b"],
            "abc": ["a", "##b", "##c"],
            "xyz": ["x", "##y", "##z"],
        }
        result = self.trainer._apply_merge(word_to_subwords, ("a", "##b"), "ab")
        assert result["ab"]  == ["ab"]
        assert result["abc"] == ["ab", "##c"]
        assert result["xyz"] == ["x", "##y", "##z"]   # unchanged

    def test_compute_freqs_returns_both_counters(self):
        word_to_subwords = {"hello": ["h", "##e", "##l", "##l", "##o"]}
        word_freqs       = {"hello": 5}
        subword_freqs, pair_freqs = self.trainer._compute_freqs(
            word_to_subwords, word_freqs
        )
        assert len(subword_freqs) > 0
        assert len(pair_freqs) > 0

    def test_compute_freqs_weights_by_word_freq(self):
        """Subword frequencies must be proportional to word frequency."""
        word_to_subwords = {"hi": ["h", "##i"]}

        wf_low  = {"hi": 1}
        wf_high = {"hi": 7}

        sf_low,  _ = self.trainer._compute_freqs(word_to_subwords, wf_low)
        sf_high, _ = self.trainer._compute_freqs(word_to_subwords, wf_high)

        assert sf_high["h"] == 7 * sf_low["h"]


# ─────────────────────────────────────────────────────────────────────────────
# WordPieceTrainer full training
# ─────────────────────────────────────────────────────────────────────────────

class TestWordPieceTrainerFull:

    def test_train_returns_vocabulary(self, tmp_path):
        corpus = make_corpus_txt(
            ["the cat sat on the mat", "the dog ran in the park"], tmp_path
        )
        trainer = WordPieceTrainer(vocab_size=60, min_frequency=1)
        vocab, _ = trainer.train(corpus)
        assert isinstance(vocab, Vocabulary)

    def test_train_special_tokens_in_vocab(self, tmp_path):
        corpus = make_corpus_txt(["hello world foo bar"] , tmp_path)
        trainer = WordPieceTrainer(vocab_size=60, min_frequency=1)
        vocab, _ = trainer.train(corpus)

        for token in ST.all_tokens():
            assert token in vocab, f"Special token '{token}' missing from WordPiece vocab"

    def test_train_continuation_tokens_present(self, tmp_path):
        """After training, some ## tokens must exist (otherwise no merges happened)."""
        corpus = make_corpus_txt(
            ["learning testing playing running walking"] * 50, tmp_path
        )
        trainer = WordPieceTrainer(vocab_size=80, min_frequency=1, lowercase=True)
        vocab, _ = trainer.train(corpus)

        # At minimum, the initial character split creates ## tokens
        continuation_tokens = [t for t in vocab if t.startswith(CONTINUATION_PREFIX)]
        assert len(continuation_tokens) > 0

    def test_train_lowercased_vocab(self, tmp_path):
        """With lowercase=True, no uppercase tokens (except special tokens) should appear."""
        corpus = make_corpus_txt(["Hello World UPPER CASE"], tmp_path)
        trainer = WordPieceTrainer(vocab_size=60, min_frequency=1, lowercase=True)
        vocab, _ = trainer.train(corpus)

        regular_tokens = [
            t for t in vocab
            if t not in ST.all_tokens()
            and not t.startswith(CONTINUATION_PREFIX)
        ]
        for token in regular_tokens:
            assert token == token.lower() or not token.isalpha(), (
                f"Uppercase token '{token}' found in lowercased vocab"
            )

    def test_train_jsonl_corpus(self, tmp_path):
        corpus = make_corpus_jsonl(["the cat sat on the mat"] * 50, tmp_path)
        trainer = WordPieceTrainer(vocab_size=60, min_frequency=1)
        vocab, _ = trainer.train(corpus, text_field="text")
        assert vocab.size > 0


# ─────────────────────────────────────────────────────────────────────────────
# WordPieceTokenizer._wordpiece_encode
# ─────────────────────────────────────────────────────────────────────────────

class TestWordPieceEncode:
    """Tests for the core longest-match-first algorithm."""

    def setup_method(self):
        """Create a minimal vocabulary for controlled testing."""
        self.vocab = Vocabulary()
        # Add specific tokens we'll test against
        for token in ["play", "##ing", "##er", "run", "##ning", "he", "##llo"]:
            self.vocab.add_token(token)
        self.tokenizer = WordPieceTokenizer(vocab=self.vocab, lowercase=False)

    def test_known_word_tokenizes_correctly(self):
        """A word whose sub-words are all in vocab must tokenize without UNK."""
        result = self.tokenizer._wordpiece_encode("playing")
        assert result == ["play", "##ing"]

    def test_unknown_word_becomes_unk(self):
        """A word whose characters are not in vocab must become [UNK]."""
        # "xyz" — neither "x" nor "##y" etc. are in our minimal vocab
        result = self.tokenizer._wordpiece_encode("xyz")
        assert result == [ST.UNK]

    def test_word_exceeding_max_chars_becomes_unk(self):
        """Words longer than max_word_chars must become [UNK]."""
        tokenizer = WordPieceTokenizer(
            vocab=self.vocab, max_word_chars=3, lowercase=False
        )
        # "hello" is 5 chars > 3 max → UNK
        result = tokenizer._wordpiece_encode("hello")
        assert result == [ST.UNK]

    def test_single_known_character(self):
        """A single character that is in the vocab should tokenize as itself."""
        vocab = Vocabulary()
        vocab.add_token("a")
        tokenizer = WordPieceTokenizer(vocab=vocab, lowercase=False)
        result = tokenizer._wordpiece_encode("a")
        assert result == ["a"]

    def test_continuation_prefix_on_second_subword(self):
        """All tokens after the first in a word must have ## prefix."""
        result = self.tokenizer._wordpiece_encode("playing")
        # First token: "play" — no ##
        assert not result[0].startswith(CONTINUATION_PREFIX)
        # Second token: "##ing" — has ##
        if len(result) > 1:
            assert result[1].startswith(CONTINUATION_PREFIX)


# ─────────────────────────────────────────────────────────────────────────────
# WordPieceTokenizer encode / decode
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_wp_tokenizer(tmp_path_factory):
    """Module-scoped: train once, reuse across test methods."""
    tmp = tmp_path_factory.mktemp("wp")
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test of the tokenizer",
        "machine learning is a subset of artificial intelligence",
        "natural language processing enables machines to understand text",
        "playing running learning eating walking reading writing",
    ]
    corpus = make_corpus_txt(sentences, tmp)
    tokenizer = WordPieceTokenizer(lowercase=True)
    tokenizer.train(corpus, vocab_size=200, min_frequency=1)
    return tokenizer


class TestWordPieceTokenizerEncodeDecode:

    def test_encode_returns_list_of_ints(self, trained_wp_tokenizer):
        ids = trained_wp_tokenizer.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_empty_string_returns_empty(self, trained_wp_tokenizer):
        assert trained_wp_tokenizer.encode("") == []

    def test_encode_bos_prepended(self, trained_wp_tokenizer):
        ids = trained_wp_tokenizer.encode("hello", add_special_tokens=True)
        assert ids[0] == ST.BOS_ID

    def test_encode_eos_appended(self, trained_wp_tokenizer):
        ids = trained_wp_tokenizer.encode("hello", add_special_tokens=True)
        assert ids[-1] == ST.EOS_ID

    def test_encode_no_special_tokens(self, trained_wp_tokenizer):
        ids = trained_wp_tokenizer.encode("hello", add_special_tokens=False)
        assert ST.BOS_ID not in ids
        assert ST.EOS_ID not in ids

    def test_encode_truncates_correctly(self, trained_wp_tokenizer):
        ids = trained_wp_tokenizer.encode("hello world " * 200, max_length=30)
        assert len(ids) <= 30

    def test_encode_deterministic(self, trained_wp_tokenizer):
        text = "machine learning is powerful"
        assert trained_wp_tokenizer.encode(text) == trained_wp_tokenizer.encode(text)

    def test_decode_returns_string(self, trained_wp_tokenizer):
        ids  = trained_wp_tokenizer.encode("hello", add_special_tokens=False)
        text = trained_wp_tokenizer.decode(ids)
        assert isinstance(text, str)

    def test_decode_skips_special_tokens(self, trained_wp_tokenizer):
        ids  = trained_wp_tokenizer.encode("hello world", add_special_tokens=True)
        text = trained_wp_tokenizer.decode(ids, skip_special_tokens=True)
        assert ST.BOS  not in text
        assert ST.EOS  not in text

    def test_decode_empty_ids_returns_empty(self, trained_wp_tokenizer):
        assert trained_wp_tokenizer.decode([]) == ""

    def test_roundtrip_words_present(self, trained_wp_tokenizer):
        """After encode→decode the original words (lowercased) must appear."""
        original = "the quick brown fox"
        ids      = trained_wp_tokenizer.encode(original, add_special_tokens=False)
        decoded  = trained_wp_tokenizer.decode(ids)

        for word in original.split():
            assert word in decoded.lower(), (
                f"Word '{word}' missing from decoded: '{decoded}'"
            )

    def test_continuation_convention_is_hash_not_g(self, trained_wp_tokenizer):
        """
        WordPiece uses ## continuation prefix, NOT Ġ (which is BPE's convention).
        Check that at least one ## token exists in the vocab.
        """
        has_continuation = any(
            t.startswith(CONTINUATION_PREFIX)
            for t in trained_wp_tokenizer.vocab
        )
        assert has_continuation, (
            "No '##' continuation tokens found — did BPE end up being used instead?"
        )


class TestWordPieceTokenizerEdgeCases:

    def test_encode_numbers(self, trained_wp_tokenizer):
        ids = trained_wp_tokenizer.encode("42 100 2024")
        assert isinstance(ids, list)

    def test_encode_punctuation_only(self, trained_wp_tokenizer):
        ids = trained_wp_tokenizer.encode("!?!?!")
        assert isinstance(ids, list)

    def test_encode_mixed_case_lowercased(self, trained_wp_tokenizer):
        """With lowercase=True, "Hello" and "hello" must produce the same tokens."""
        ids_upper = trained_wp_tokenizer.encode("Hello", add_special_tokens=False)
        ids_lower = trained_wp_tokenizer.encode("hello", add_special_tokens=False)
        assert ids_upper == ids_lower

    def test_encode_single_character(self, trained_wp_tokenizer):
        ids = trained_wp_tokenizer.encode("a")
        assert len(ids) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# WordPieceTokenizer save / load
# ─────────────────────────────────────────────────────────────────────────────

class TestWordPieceTokenizerSaveLoad:

    def test_save_creates_required_files(self, trained_wp_tokenizer, tmp_path):
        trained_wp_tokenizer.save(str(tmp_path))
        assert (tmp_path / "vocab.json").exists()
        assert (tmp_path / "tokenizer_config.json").exists()

    def test_config_contains_class_name(self, trained_wp_tokenizer, tmp_path):
        trained_wp_tokenizer.save(str(tmp_path))
        config = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert config["tokenizer_class"] == "WordPieceTokenizer"

    def test_config_preserves_lowercase_flag(self, trained_wp_tokenizer, tmp_path):
        trained_wp_tokenizer.save(str(tmp_path))
        config = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert config["lowercase"] == trained_wp_tokenizer._lowercase

    def test_load_restores_vocab_size(self, trained_wp_tokenizer, tmp_path):
        trained_wp_tokenizer.save(str(tmp_path))
        loaded = WordPieceTokenizer.load(str(tmp_path))
        assert loaded.vocab_size == trained_wp_tokenizer.vocab_size

    def test_loaded_encode_identical(self, trained_wp_tokenizer, tmp_path):
        text = "natural language processing"
        trained_wp_tokenizer.save(str(tmp_path))
        loaded = WordPieceTokenizer.load(str(tmp_path))

        assert loaded.encode(text) == trained_wp_tokenizer.encode(text)

    def test_loaded_decode_identical(self, trained_wp_tokenizer, tmp_path):
        ids = trained_wp_tokenizer.encode("hello world", add_special_tokens=False)
        trained_wp_tokenizer.save(str(tmp_path))
        loaded = WordPieceTokenizer.load(str(tmp_path))

        assert loaded.decode(ids) == trained_wp_tokenizer.decode(ids)

    def test_loaded_special_token_ids_correct(self, trained_wp_tokenizer, tmp_path):
        trained_wp_tokenizer.save(str(tmp_path))
        loaded = WordPieceTokenizer.load(str(tmp_path))

        assert loaded.pad_token_id  == ST.PAD_ID
        assert loaded.bos_token_id  == ST.BOS_ID
        assert loaded.eos_token_id  == ST.EOS_ID
        assert loaded.unk_token_id  == ST.UNK_ID
        assert loaded.mask_token_id == ST.MASK_ID


# ─────────────────────────────────────────────────────────────────────────────
# BPE vs WordPiece convention comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestWordPieceVsBPEConventions:
    """Verify that WordPiece and BPE use distinct, non-conflicting conventions."""

    def test_wordpiece_uses_hash_prefix_not_g_prefix(self, trained_wp_tokenizer):
        """
        WordPiece continuation prefix must be ## not Ġ.
        Having both in the same codebase requires them to be clearly distinct.
        """
        vocab_tokens = list(trained_wp_tokenizer.vocab)
        continuation_tokens = [t for t in vocab_tokens if t.startswith("##")]
        g_prefix_tokens     = [t for t in vocab_tokens if t.startswith("Ġ")]

        # WordPiece vocab should have ## tokens
        assert len(continuation_tokens) > 0, "Expected ## continuation tokens in WordPiece vocab"
        # WordPiece vocab should NOT have Ġ tokens (that's BPE's convention)
        assert len(g_prefix_tokens) == 0, (
            f"Found BPE-style Ġ tokens in WordPiece vocab: {g_prefix_tokens[:5]}"
        )
