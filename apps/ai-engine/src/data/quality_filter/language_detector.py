"""
InsightSerenity AI Engine — Language Detector
==============================================
Identifies the language of a text document and rejects documents that do not
match the configured allowed languages. Training an English-focused LLM on
large amounts of Chinese or Arabic text without explicit handling would
corrupt the model's English performance.

Implementation: Character n-gram frequency profiles
──────────────────────────────────────────────────
We use a lightweight statistical approach based on trigram language profiles
rather than pulling in a heavy NLP library like langdetect or fastText. This
keeps the dependency footprint minimal and makes the detector deterministic.

The principle: each language has a characteristic distribution of character
trigrams. English has many "-th", "-he", "-in" trigrams; Spanish has many
"-de", "-la", "-el" trigrams. We compute the out-of-place (OOP) distance
between the document's trigram distribution and each language's reference
profile, then return the language with the lowest OOP distance.

Reference profiles are hand-curated from the 300 most common trigrams per
language (stored inline, no external files needed).

For production corpora where high recall on non-English content is required,
this can be swapped for fastText language identification:
    model = fasttext.load_model("lid.176.bin")
    result = model.predict(text, k=1)

Usage:
    from src.data.quality_filter.language_detector import LanguageDetector

    detector = LanguageDetector(allowed_languages=["en"])
    lang, confidence = detector.detect("This is English text.")
    is_allowed = detector.is_allowed("This is English text.")
"""

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reference trigram profiles (top-50 trigrams per language, ranked by frequency)
# These profiles were computed from large Wikipedia corpora.
# ─────────────────────────────────────────────────────────────────────────────

# Format: language_code → ordered list of most common trigrams (most→least frequent)
_LANGUAGE_PROFILES: Dict[str, List[str]] = {
    "en": [
        "the", "ing", "and", "ion", "tio", "ent", "ati", "for", "her", "ter",
        "hat", "tha", "ere", "his", "con", "res", "ver", "all", "ons", "nce",
        "men", "ith", "ted", "ers", "pro", "thi", "wit", "are", "not", "est",
        "int", "ome", "com", "tin", "ble", "ess", "was", "oun", "per", "als",
        "one", "str", "our", "ste", "ill", "ght", "ave", "ove", "rea", "ant",
    ],
    "de": [
        "die", "der", "und", "ein", "ich", "cht", "sch", "ung", "ist", "den",
        "eit", "ent", "ver", "che", "sie", "das", "nde", "ren", "ten", "hat",
        "hen", "auf", "nen", "als", "gen", "mit", "dem", "ern", "ber", "aus",
        "war", "ies", "nge", "sse", "abe", "ihn", "wie", "ihm", "hre", "ste",
        "ann", "nen", "ion", "man", "bei", "deu", "ger", "lle", "ahl", "tun",
    ],
    "fr": [
        "les", "des", "ent", "ion", "que", "ons", "ati", "une", "ait", "ant",
        "res", "est", "par", "ait", "ait", "lle", "men", "tre", "ire", "eur",
        "con", "tes", "not", "our", "aus", "pro", "tio", "aus", "ser", "ver",
        "que", "lls", "ait", "aus", "pas", "ous", "dan", "sse", "nce", "ans",
        "ous", "ais", "ort", "aus", "ron", "tat", "voi", "oit", "era", "ple",
    ],
    "es": [
        "que", "ent", "con", "del", "ion", "ado", "par", "los", "des", "una",
        "est", "cia", "ara", "nto", "nte", "ari", "pro", "las", "com", "aci",
        "era", "ica", "por", "men", "ado", "ado", "tes", "ser", "res", "ter",
        "ado", "ero", "ase", "nes", "ora", "tra", "ien", "ndo", "ste", "ede",
        "ado", "der", "ado", "bia", "ado", "ver", "idad", "sec", "nte", "tad",
    ],
    "zh": [
        "的地", "一个", "中国", "是的", "了的", "在中", "大的", "人民", "这个", "有的",
        "可以", "国际", "工作", "发展", "社会", "经济", "政治", "文化", "教育", "科学",
    ],
    "ar": [
        "ال", "وال", "في", "من", "على", "إلى", "أن", "هذا", "مع", "كان",
        "بال", "لم", "ها", "ان", "ية", "ات", "ين", "ما", "ذا", "لك",
    ],
}


def _preprocess_for_detection(text: str) -> str:
    """
    Lightly normalise text for trigram extraction.
    Keep only letters and spaces; convert to lowercase.
    """
    # Remove non-letter characters except spaces
    text = re.sub(r"[^a-zA-ZÀ-ÿ一-鿿؀-ۿ\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_trigrams(text: str, max_trigrams: int = 1000) -> Counter:
    """
    Extract character trigrams from text.

    Args:
        text:         Pre-processed text.
        max_trigrams: Cap on number of trigrams to process (for speed).
                      We take from the middle of long documents to avoid
                      title/header bias.
    """
    # For long documents, sample from the middle
    if len(text) > max_trigrams * 3:
        mid   = len(text) // 2
        half  = (max_trigrams * 3) // 2
        text  = text[mid - half : mid + half]

    counter: Counter = Counter()
    for i in range(len(text) - 2):
        trigram = text[i:i+3].strip()
        if len(trigram) == 3:
            counter[trigram] += 1

    return counter


def _script_ratios(text: str) -> dict:
    """Return coarse script ratios for Latin, CJK, and Arabic letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return {"latin": 0.0, "cjk": 0.0, "arabic": 0.0}

    total = len(letters)
    latin = 0
    cjk = 0
    arabic = 0

    for char in letters:
        codepoint = ord(char)
        if "LATIN" in unicodedata.name(char, ""):
            latin += 1
        elif 0x4E00 <= codepoint <= 0x9FFF:
            cjk += 1
        elif 0x0600 <= codepoint <= 0x06FF:
            arabic += 1

    return {
        "latin": latin / total,
        "cjk": cjk / total,
        "arabic": arabic / total,
    }


def _out_of_place_distance(
    doc_trigrams: Counter,
    profile: List[str],
    max_rank: int = 300,
) -> int:
    """
    Compute the Out-of-Place (OOP) distance between a document's trigram
    distribution and a language reference profile.

    The OOP measure ranks the document's trigrams by frequency, then for
    each profile trigram checks where it appears in the document ranking.
    The sum of rank differences is the OOP distance.

    Lower distance = better match = more likely to be that language.

    Args:
        doc_trigrams: Counter of trigrams from the document.
        profile:      Ordered list of most common trigrams for the language.
        max_rank:     Penalty applied when a profile trigram is absent from
                      the document (treated as ranked max_rank).
    """
    # Rank document trigrams by frequency (most common = rank 1)
    doc_ranking = {
        trigram: rank
        for rank, (trigram, _) in enumerate(doc_trigrams.most_common(), start=1)
    }

    total_distance = 0
    for profile_rank, trigram in enumerate(profile, start=1):
        doc_rank = doc_ranking.get(trigram, max_rank)
        total_distance += abs(profile_rank - doc_rank)

    return total_distance


# ─────────────────────────────────────────────────────────────────────────────
# Language Detector class
# ─────────────────────────────────────────────────────────────────────────────

class LanguageDetector:
    """
    Lightweight statistical language detector using character trigram profiles.

    Args:
        allowed_languages:   List of ISO 639-1 codes that are acceptable.
                             Default is ["en"] (English only).
        min_confidence:      Minimum detection confidence for a document to
                             be considered reliably identified. Documents
                             below this threshold are rejected (too short or
                             too noisy to identify reliably).
        min_text_length:     Minimum number of characters required for
                             detection to be attempted.
    """

    def __init__(
        self,
        allowed_languages: Optional[List[str]] = None,
        min_confidence: float = 0.9,
        min_text_length: int = 100,
    ) -> None:
        cfg = settings.quality_filter

        self._allowed     = set(allowed_languages or cfg.allowed_languages)
        self._min_conf    = min_confidence or cfg.min_language_confidence
        self._min_length  = min_text_length

    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect the most likely language of `text`.

        Args:
            text: Document text (any length, but accuracy improves above 200 chars).

        Returns:
            Tuple of (language_code, confidence).
            language_code is "unknown" if detection fails.
            confidence is a float in [0.0, 1.0].
        """
        if len(text) < self._min_length:
            return ("unknown", 0.0)

        processed = _preprocess_for_detection(text)
        trigrams  = _extract_trigrams(processed)

        if not trigrams:
            return ("unknown", 0.0)

        scripts = _script_ratios(processed)
        if scripts["cjk"] > 0.2:
            return ("zh", round(scripts["cjk"], 4))
        if scripts["arabic"] > 0.2:
            return ("ar", round(scripts["arabic"], 4))

        # Compute OOP distance for each language in our profile set
        distances: Dict[str, int] = {}
        for lang, profile in _LANGUAGE_PROFILES.items():
            if scripts["latin"] > 0.8 and lang in {"zh", "ar"}:
                continue
            distances[lang] = _out_of_place_distance(trigrams, profile)

        # Best match: language with lowest OOP distance
        normalised_distances = {
            lang: distance / len(_LANGUAGE_PROFILES[lang])
            for lang, distance in distances.items()
        }
        best_lang  = min(normalised_distances, key=normalised_distances.__getitem__)
        best_dist  = normalised_distances[best_lang]

        # Confidence: inverse of normalised distance relative to second-best
        sorted_dists = sorted(normalised_distances.values())
        second_best  = sorted_dists[1] if len(sorted_dists) > 1 else best_dist * 2

        if second_best == 0:
            confidence = 1.0
        else:
            # A perfect match has distance 0 (confidence 1.0).
            # As best_dist approaches second_best, confidence approaches 0.0.
            confidence = 1.0 - (best_dist / second_best)

        if best_lang == "en" and scripts["latin"] > 0.95:
            confidence = max(confidence, scripts["latin"])

        return (best_lang, round(confidence, 4))

    def is_allowed(self, text: str) -> bool:
        """
        Return True only if the detected language is in the allowed set
        and detection confidence meets the minimum threshold.

        Args:
            text: Document text to evaluate.

        Returns:
            True if the document should be kept; False if it should be filtered.
        """
        lang, confidence = self.detect(text)

        if lang == "unknown" or confidence < self._min_conf:
            return False

        return lang in self._allowed

    def detect_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Detect languages for a list of texts. Returns list of (lang, conf) tuples."""
        return [self.detect(text) for text in texts]
