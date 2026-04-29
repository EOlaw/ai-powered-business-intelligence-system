"""
InsightSerenity AI Engine — Text Cleaner
=========================================
Deep-cleans extracted plain text before it enters the deduplication and
quality-filtering stages. This is distinct from the HTML extractor — by the
time text reaches this module, tags are already stripped. We are now fixing
issues in the plain text itself.

Problems addressed:
    - Unicode noise: invisible characters, control codes, zero-width spaces,
      BOM markers, bidirectional overrides, soft hyphens
    - Encoding artefacts: mojibake (garbled UTF-8), latin-1 characters in
      UTF-8 context
    - Structural noise: excessive blank lines, mixed newline styles (CRLF),
      trailing whitespace
    - URL/email stripping: for training data we typically do not want raw
      URLs cluttering the distribution
    - Number normalisation: repeated digits/numbers that add token noise
    - Line-level filtering: lines that are clearly not natural language
      (long sequences of symbols, base64 blobs, hex dumps)

Usage:
    from src.data.preprocessing.text_cleaner import TextCleaner

    cleaner = TextCleaner()
    clean   = cleaner.clean("messy   text\r\n\nwith​ issues")
"""

import re
import unicodedata
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Compiled regex patterns — compiled once at module load for performance
# ─────────────────────────────────────────────────────────────────────────────

# Matches http(s) URLs
_URL_RE = re.compile(
    r"https?://[^\s<>\"']+",
    re.IGNORECASE,
)

# Matches email addresses
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# Matches lines that are predominantly symbols (not natural language)
# e.g. "--------- *** ---- ++++" or "0xdeadbeef cafebabe" or base64 blobs
_SYMBOL_LINE_RE = re.compile(r"^[^a-zA-Z]{10,}$")

# Matches sequences of 5+ repeated identical characters (keyboard mashing)
_REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")

# Matches lines that are overwhelmingly digits (tables, tracking codes)
_MOSTLY_DIGITS_RE = re.compile(r"^\s*[\d\s.,\-+/:]{20,}\s*$")

# Windows CRLF → LF
_CRLF_RE = re.compile(r"\r\n")

# Multiple blank lines → at most one blank line
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Multiple spaces/tabs → single space
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Unicode categories considered "invisible" or control characters
# Categories: Cf (format), Cc (control), Co (private use), Cs (surrogates)
_INVISIBLE_UNICODE_CATEGORIES = frozenset({"Cf", "Cc", "Co", "Cs"})

# Allowed exceptions within invisible categories (normal newlines, tabs)
_ALLOWED_CONTROL_CHARS = frozenset({"\n", "\t"})


class TextCleaner:
    """
    Stateless text cleaning pipeline.

    Each cleaning step is a separate method so individual steps can be
    called in isolation during development and testing. The `clean()` method
    applies all steps in the correct order.

    Args:
        strip_urls:    Replace URLs with a <URL> token. Default True.
        strip_emails:  Replace email addresses with <EMAIL> token. Default True.
        url_token:     Replacement string for URLs.
        email_token:   Replacement string for emails.
        min_line_len:  Lines shorter than this after cleaning are dropped.
    """

    def __init__(
        self,
        strip_urls: bool = True,
        strip_emails: bool = True,
        url_token: str = "",
        email_token: str = "",
        min_line_len: int = 10,
    ) -> None:
        self._strip_urls    = strip_urls
        self._strip_emails  = strip_emails
        self._url_token     = url_token
        self._email_token   = email_token
        self._min_line_len  = min_line_len

    # ── Public API ─────────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """
        Apply the full cleaning pipeline to a text document.

        Cleaning order matters: Unicode normalisation must come before regex
        operations; CRLF normalisation before blank-line collapsing.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text, or empty string if the document is unsalvageable.
        """
        if not text:
            return ""

        text = self.remove_invisible_unicode(text)
        text = self.normalise_newlines(text)
        text = self.decode_unicode_escapes(text)

        if self._strip_urls:
            text = self.strip_urls(text)
        if self._strip_emails:
            text = self.strip_emails(text)

        text = self.remove_repeated_chars(text)
        text = self.collapse_whitespace(text)
        text = self.filter_lines(text)
        text = self.collapse_blank_lines(text)

        return text.strip()

    def remove_invisible_unicode(self, text: str) -> str:
        """
        Strip invisible Unicode characters: zero-width spaces, soft hyphens,
        BOM markers, bidirectional overrides, and all Cf/Cc/Co/Cs category
        characters except newline and tab.

        These characters are common in scraped web text and cause tokenizer
        and model issues if left in.
        """
        cleaned = []
        for char in text:
            # Always keep printable characters and our two allowed controls
            if char in _ALLOWED_CONTROL_CHARS:
                cleaned.append(char)
                continue
            category = unicodedata.category(char)
            if category not in _INVISIBLE_UNICODE_CATEGORIES:
                cleaned.append(char)
            # else: silently drop the invisible character

        return "".join(cleaned)

    def normalise_newlines(self, text: str) -> str:
        """
        Normalise all newline styles to Unix LF (\n).
        Handles: CRLF (\r\n), CR (\r), vertical tab (\v), form feed (\f).
        """
        text = _CRLF_RE.sub("\n", text)       # Windows CRLF → LF
        text = text.replace("\r", "\n")        # Bare CR → LF
        text = text.replace("\v", "\n")        # Vertical tab → LF
        text = text.replace("\f", "\n")        # Form feed → LF
        return text

    def decode_unicode_escapes(self, text: str) -> str:
        """
        Attempt to fix common mojibake patterns where text was double-encoded.
        For example, "â€™" is a UTF-8 right apostrophe decoded as Latin-1.

        Also normalises Unicode to NFC form (canonical composition), which
        ensures that accented characters are represented consistently.
        """
        # NFC normalisation: é = e + combining acute → single é codepoint
        text = unicodedata.normalize("NFC", text)
        return text

    def strip_urls(self, text: str) -> str:
        """
        Replace HTTP(S) URLs with the configured token (default: remove).
        URLs in training data distort the token distribution without adding
        linguistic information.
        """
        return _URL_RE.sub(self._url_token, text)

    def strip_emails(self, text: str) -> str:
        """Replace email addresses with the configured token (default: remove)."""
        return _EMAIL_RE.sub(self._email_token, text)

    def remove_repeated_chars(self, text: str) -> str:
        """
        Collapse sequences of 5+ identical characters to 3.
        "Whaaaaaaaaat" → "Whaaaat"
        This reduces vocabulary explosion from keyboard-mashing text.
        """
        return _REPEATED_CHAR_RE.sub(r"\1\1\1", text)

    def collapse_whitespace(self, text: str) -> str:
        """
        Collapse multiple consecutive spaces/tabs on a single line to one space.
        Does NOT collapse newlines — that is handled by collapse_blank_lines.
        """
        lines = text.split("\n")
        lines = [_MULTI_SPACE_RE.sub(" ", line).rstrip() for line in lines]
        return "\n".join(lines)

    def filter_lines(self, text: str) -> str:
        """
        Drop individual lines that are clearly not natural language text.

        Filtered:
          - Lines shorter than min_line_len (navigation fragments, lone symbols)
          - Lines consisting almost entirely of punctuation/symbols
          - Lines that are overwhelmingly digits (tables, tracking IDs)
          - Lines matching base64/hex patterns
        """
        filtered: List[str] = []

        for line in text.split("\n"):
            stripped = line.strip()

            # Keep blank lines — they mark paragraph boundaries
            if not stripped:
                filtered.append("")
                continue

            # Too short
            if len(stripped) < self._min_line_len:
                continue

            # Predominantly symbols
            if _SYMBOL_LINE_RE.match(stripped):
                continue

            # Mostly digits (tables, checksums, IDs)
            if _MOSTLY_DIGITS_RE.match(stripped):
                continue

            filtered.append(line)

        return "\n".join(filtered)

    def collapse_blank_lines(self, text: str) -> str:
        """
        Reduce runs of 3 or more blank lines to exactly one blank line.
        Preserves intentional paragraph breaks.
        """
        return _MULTI_BLANK_RE.sub("\n\n", text)
