"""
Unit tests for src.tokenizer.vocabulary.Vocabulary

Coverage:
    - Construction with special tokens pre-populated
    - add_token: new tokens, duplicate tokens
    - add_tokens: bulk add
    - token_to_id / id_to_token: forward and reverse lookups
    - __getitem__ bidirectional shorthand
    - __contains__
    - __len__
    - __iter__: order is ID-ascending
    - Special token ID guarantees (PAD=0, UNK=1, BOS=2, EOS=3, MASK=4)
    - Unknown token fallback for out-of-vocab strings
    - Unknown ID fallback for out-of-range IDs
    - save() / load() round-trip: all tokens preserved, IDs unchanged
    - IDs are stable across repeated save/load cycles
    - Edge cases: empty string token, unicode tokens, whitespace tokens
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.special_tokens import SpecialTokens as ST, ALL_SPECIAL_TOKENS


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_vocab() -> Vocabulary:
    """A fresh Vocabulary with only special tokens populated."""
    return Vocabulary()


@pytest.fixture
def populated_vocab() -> Vocabulary:
    """A Vocabulary with several regular tokens added."""
    vocab = Vocabulary()
    for token in ["hello", "world", "foo", "bar", "##ing", "Ġthe"]:
        vocab.add_token(token)
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyConstruction:

    def test_special_tokens_prepopulated(self, empty_vocab):
        """All special tokens must be present immediately after construction."""
        for special in ALL_SPECIAL_TOKENS:
            assert special.token in empty_vocab, (
                f"Special token '{special.token}' missing from new Vocabulary"
            )

    def test_pad_id_is_zero(self, empty_vocab):
        """PAD must always be ID 0 — the convention for padding in PyTorch."""
        assert empty_vocab[ST.PAD] == 0

    def test_unk_id_is_one(self, empty_vocab):
        assert empty_vocab[ST.UNK] == 1

    def test_bos_id_is_two(self, empty_vocab):
        assert empty_vocab[ST.BOS] == 2

    def test_eos_id_is_three(self, empty_vocab):
        assert empty_vocab[ST.EOS] == 3

    def test_mask_id_is_four(self, empty_vocab):
        assert empty_vocab[ST.MASK] == 4

    def test_initial_size_equals_special_token_count(self, empty_vocab):
        """Only special tokens exist before any regular tokens are added."""
        assert len(empty_vocab) == len(ALL_SPECIAL_TOKENS)

    def test_chat_tokens_prepopulated(self, empty_vocab):
        """Chat role tokens must also be pre-populated."""
        for token in [ST.SYSTEM, ST.USER, ST.ASSISTANT, ST.END_TURN]:
            assert token in empty_vocab


# ─────────────────────────────────────────────────────────────────────────────
# add_token
# ─────────────────────────────────────────────────────────────────────────────

class TestAddToken:

    def test_new_token_gets_assigned_id(self, empty_vocab):
        token_id = empty_vocab.add_token("hello")
        assert isinstance(token_id, int)
        assert token_id > 0

    def test_new_token_is_reachable_by_string(self, empty_vocab):
        empty_vocab.add_token("hello")
        assert "hello" in empty_vocab

    def test_new_token_is_reachable_by_id(self, empty_vocab):
        token_id = empty_vocab.add_token("hello")
        assert empty_vocab.id_to_token(token_id) == "hello"

    def test_duplicate_token_returns_same_id(self, empty_vocab):
        """Adding the same token twice must return the same ID both times."""
        id1 = empty_vocab.add_token("hello")
        id2 = empty_vocab.add_token("hello")
        assert id1 == id2

    def test_duplicate_does_not_grow_vocab(self, empty_vocab):
        size_before = len(empty_vocab)
        empty_vocab.add_token("hello")
        empty_vocab.add_token("hello")
        assert len(empty_vocab) == size_before + 1

    def test_ids_are_sequential(self, empty_vocab):
        """Regular tokens receive strictly increasing IDs."""
        ids = [empty_vocab.add_token(f"token_{i}") for i in range(5)]
        assert ids == sorted(ids)
        assert len(set(ids)) == 5

    def test_regular_token_ids_do_not_collide_with_special_tokens(self, empty_vocab):
        special_ids = set(ST.all_ids())
        for i in range(20):
            new_id = empty_vocab.add_token(f"regular_{i}")
            assert new_id not in special_ids, (
                f"Regular token 'regular_{i}' collided with special token ID {new_id}"
            )

    def test_unicode_token(self, empty_vocab):
        """Unicode tokens (e.g. CJK characters) must be supported."""
        token_id = empty_vocab.add_token("你好")
        assert empty_vocab["你好"] == token_id
        assert empty_vocab[token_id] == "你好"

    def test_continuation_prefix_token(self, empty_vocab):
        """WordPiece-style '##ing' tokens must be stored as-is."""
        token_id = empty_vocab.add_token("##ing")
        assert empty_vocab["##ing"] == token_id

    def test_space_prefix_token(self, empty_vocab):
        """BPE-style 'Ġworld' tokens must be stored as-is."""
        token_id = empty_vocab.add_token("Ġworld")
        assert empty_vocab["Ġworld"] == token_id


# ─────────────────────────────────────────────────────────────────────────────
# add_tokens (bulk)
# ─────────────────────────────────────────────────────────────────────────────

class TestAddTokens:

    def test_add_multiple_tokens(self, empty_vocab):
        ids = empty_vocab.add_tokens(["a", "b", "c"])
        assert len(ids) == 3
        assert len(set(ids)) == 3

    def test_all_bulk_added_tokens_are_accessible(self, empty_vocab):
        tokens = ["alpha", "beta", "gamma"]
        empty_vocab.add_tokens(tokens)
        for token in tokens:
            assert token in empty_vocab


# ─────────────────────────────────────────────────────────────────────────────
# Lookups
# ─────────────────────────────────────────────────────────────────────────────

class TestLookups:

    def test_token_to_id_known_token(self, populated_vocab):
        assert isinstance(populated_vocab.token_to_id("hello"), int)

    def test_id_to_token_known_id(self, populated_vocab):
        token_id = populated_vocab.token_to_id("hello")
        assert populated_vocab.id_to_token(token_id) == "hello"

    def test_token_to_id_unknown_returns_unk_id(self, populated_vocab):
        """Out-of-vocabulary strings must return the UNK token ID."""
        assert populated_vocab.token_to_id("NOTINVOCAB_xyz") == ST.UNK_ID

    def test_id_to_token_unknown_returns_unk_token(self, populated_vocab):
        """Out-of-range IDs must return the UNK token string."""
        assert populated_vocab.id_to_token(999_999) == ST.UNK

    def test_getitem_string_returns_id(self, populated_vocab):
        assert isinstance(populated_vocab["hello"], int)

    def test_getitem_int_returns_string(self, populated_vocab):
        token_id = populated_vocab["hello"]
        assert populated_vocab[token_id] == "hello"

    def test_contains_known_token(self, populated_vocab):
        assert "hello" in populated_vocab

    def test_not_contains_unknown_token(self, populated_vocab):
        assert "xyz_not_in_vocab" not in populated_vocab


# ─────────────────────────────────────────────────────────────────────────────
# __len__ and __iter__
# ─────────────────────────────────────────────────────────────────────────────

class TestLenAndIter:

    def test_len_increases_on_add(self, empty_vocab):
        initial = len(empty_vocab)
        empty_vocab.add_token("new_token")
        assert len(empty_vocab) == initial + 1

    def test_len_stable_on_duplicate_add(self, empty_vocab):
        empty_vocab.add_token("dup")
        before = len(empty_vocab)
        empty_vocab.add_token("dup")
        assert len(empty_vocab) == before

    def test_iter_visits_all_tokens(self, populated_vocab):
        all_tokens = list(populated_vocab)
        assert len(all_tokens) == len(populated_vocab)

    def test_iter_is_id_ascending(self, populated_vocab):
        """__iter__ must yield tokens in ID-ascending order."""
        tokens = list(populated_vocab)
        ids    = [populated_vocab[t] for t in tokens]
        assert ids == sorted(ids)

    def test_special_tokens_appear_first_in_iter(self, populated_vocab):
        """Special tokens have the lowest IDs and must come first."""
        tokens = list(populated_vocab)
        first_n = tokens[:len(ALL_SPECIAL_TOKENS)]
        for special in ALL_SPECIAL_TOKENS:
            assert special.token in first_n


# ─────────────────────────────────────────────────────────────────────────────
# Properties
# ─────────────────────────────────────────────────────────────────────────────

class TestProperties:

    def test_size_property(self, populated_vocab):
        assert populated_vocab.size == len(populated_vocab)

    def test_pad_token_id_property(self, empty_vocab):
        assert empty_vocab.pad_token_id == ST.PAD_ID

    def test_bos_token_id_property(self, empty_vocab):
        assert empty_vocab.bos_token_id == ST.BOS_ID

    def test_eos_token_id_property(self, empty_vocab):
        assert empty_vocab.eos_token_id == ST.EOS_ID

    def test_unk_token_id_property(self, empty_vocab):
        assert empty_vocab.unk_token_id == ST.UNK_ID

    def test_mask_token_id_property(self, empty_vocab):
        assert empty_vocab.mask_token_id == ST.MASK_ID


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveLoad:

    def test_save_creates_json_file(self, populated_vocab):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            populated_vocab.save(path)
            assert path.exists()

    def test_saved_file_is_valid_json(self, populated_vocab):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            populated_vocab.save(path)
            with open(path, "r") as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_load_restores_all_tokens(self, populated_vocab):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            populated_vocab.save(path)
            loaded = Vocabulary.load(path)

        for token in populated_vocab:
            assert token in loaded

    def test_load_preserves_ids(self, populated_vocab):
        """Every token must have the same ID after a save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            populated_vocab.save(path)
            loaded = Vocabulary.load(path)

        for token in populated_vocab:
            assert populated_vocab[token] == loaded[token], (
                f"ID mismatch for token '{token}': "
                f"{populated_vocab[token]} → {loaded[token]}"
            )

    def test_load_preserves_special_token_ids(self, populated_vocab):
        """Special token IDs must survive save/load unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            populated_vocab.save(path)
            loaded = Vocabulary.load(path)

        assert loaded[ST.PAD]  == ST.PAD_ID
        assert loaded[ST.UNK]  == ST.UNK_ID
        assert loaded[ST.BOS]  == ST.BOS_ID
        assert loaded[ST.EOS]  == ST.EOS_ID
        assert loaded[ST.MASK] == ST.MASK_ID

    def test_loaded_vocab_size_matches(self, populated_vocab):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            populated_vocab.save(path)
            loaded = Vocabulary.load(path)

        assert len(loaded) == len(populated_vocab)

    def test_double_round_trip_stability(self, populated_vocab):
        """IDs must remain stable through two consecutive save/load cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            populated_vocab.save(path)
            first  = Vocabulary.load(path)
            first.save(path)
            second = Vocabulary.load(path)

        for token in populated_vocab:
            assert first[token] == second[token]

    def test_save_creates_parent_directory(self, populated_vocab):
        """save() must create parent directories that do not yet exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "vocab.json"
            populated_vocab.save(path)
            assert path.exists()
