"""
InsightSerenity AI Engine — Special Tokens
==========================================
Defines every special token used across all tokenizers in the platform.
Special tokens are tokens with specific semantic meaning that the model
learns to interpret beyond their surface text.

All tokenizer implementations must agree on these token strings and their
IDs. The IDs are assigned here at the vocabulary level — special tokens
always occupy the first N positions in the vocabulary for easy identification.

Token semantics:
    PAD   — Padding: fills sequences to uniform length. The model ignores
             these positions via the attention mask. Loss is not computed on
             PAD tokens.

    UNK   — Unknown: replaces characters/subwords not in the vocabulary.
             Well-trained BPE tokenizers should rarely produce this token.

    BOS   — Beginning of sequence: prepended to every input. Tells the model
             where each example starts.

    EOS   — End of sequence: appended to every output. The model learns to
             generate this token when it has finished.

    MASK  — Masked token: used in BERT-style masked language modeling (MLM)
             pretraining. A fraction of input tokens are replaced with MASK
             and the model predicts the original token.

    SEP   — Separator: divides two segments in classification tasks
             (e.g. question [SEP] context in question-answering).

    CLS   — Classification: prepended in BERT-style models; the CLS
             representation is used as the pooled sentence representation.

    SYSTEM / USER / ASSISTANT — Conversation role delimiters for the
             instruction-tuned and chat-fine-tuned variants of our LLM.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SpecialToken:
    """
    Represents a single special token with its string form and reserved ID.

    Args:
        token:      The string representation (e.g. "<pad>").
        token_id:   The integer ID this token is always assigned.
        description: Human-readable explanation of this token's role.
    """
    token:       str
    token_id:    int
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# Special token definitions
# All IDs are in [0, 7] — always the first 8 vocabulary positions.
# ─────────────────────────────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    SpecialToken("<pad>",       0, "Padding token — attention mask = 0"),
    SpecialToken("<unk>",       1, "Unknown token — subword not in vocabulary"),
    SpecialToken("<bos>",       2, "Beginning of sequence"),
    SpecialToken("<eos>",       3, "End of sequence"),
    SpecialToken("<mask>",      4, "Masked token for MLM pretraining"),
    SpecialToken("<sep>",       5, "Segment separator for classification tasks"),
    SpecialToken("<cls>",       6, "Classification token for sentence pooling"),
    SpecialToken("<pad_extra>", 7, "Reserved for future use"),
]

# Conversation role delimiters (for instruction-tuned model)
CHAT_TOKENS = [
    SpecialToken("<|system|>",    8,  "System prompt delimiter"),
    SpecialToken("<|user|>",      9,  "User turn delimiter"),
    SpecialToken("<|assistant|>", 10, "Assistant turn delimiter"),
    SpecialToken("<|end_turn|>",  11, "End of a conversation turn"),
]

ALL_SPECIAL_TOKENS: List[SpecialToken] = SPECIAL_TOKENS + CHAT_TOKENS


# ─────────────────────────────────────────────────────────────────────────────
# Convenient name-based access
# ─────────────────────────────────────────────────────────────────────────────

class SpecialTokens:
    """
    Namespace for programmatic access to special token strings and IDs.

    Usage:
        from src.tokenizer.special_tokens import SpecialTokens as ST

        ST.PAD          # "<pad>"
        ST.PAD_ID       # 0
        ST.all_tokens() # list of all special token strings
    """
    PAD:       str = "<pad>"
    UNK:       str = "<unk>"
    BOS:       str = "<bos>"
    EOS:       str = "<eos>"
    MASK:      str = "<mask>"
    SEP:       str = "<sep>"
    CLS:       str = "<cls>"
    SYSTEM:    str = "<|system|>"
    USER:      str = "<|user|>"
    ASSISTANT: str = "<|assistant|>"
    END_TURN:  str = "<|end_turn|>"

    PAD_ID:       int = 0
    UNK_ID:       int = 1
    BOS_ID:       int = 2
    EOS_ID:       int = 3
    MASK_ID:      int = 4
    SEP_ID:       int = 5
    CLS_ID:       int = 6
    SYSTEM_ID:    int = 8
    USER_ID:      int = 9
    ASSISTANT_ID: int = 10
    END_TURN_ID:  int = 11

    @classmethod
    def all_tokens(cls) -> List[str]:
        """Return all special token strings in ID order."""
        return [st.token for st in ALL_SPECIAL_TOKENS]

    @classmethod
    def all_ids(cls) -> List[int]:
        """Return all special token IDs."""
        return [st.token_id for st in ALL_SPECIAL_TOKENS]

    @classmethod
    def token_to_id_map(cls) -> Dict[str, int]:
        """Return a dict mapping token string → token ID."""
        return {st.token: st.token_id for st in ALL_SPECIAL_TOKENS}

    @classmethod
    def id_to_token_map(cls) -> Dict[int, str]:
        """Return a dict mapping token ID → token string."""
        return {st.token_id: st.token for st in ALL_SPECIAL_TOKENS}

    @classmethod
    def num_special_tokens(cls) -> int:
        """Total number of special tokens (including chat tokens)."""
        return len(ALL_SPECIAL_TOKENS)
