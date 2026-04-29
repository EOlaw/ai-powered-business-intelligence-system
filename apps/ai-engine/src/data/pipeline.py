"""
InsightSerenity AI Engine — Data Pipeline Orchestrator
=======================================================
The top-level coordinator for the full data acquisition and preparation
pipeline. Runs all Phase 1 stages in the correct order:

    1. CRAWL        — fetch raw HTML from seed URLs
    2. EXTRACT      — HTML → clean text (HTMLExtractor)
    3. CLEAN        — remove Unicode noise, URLs, etc. (TextCleaner)
    4. NORMALISE    — typographic normalisation (TextNormalizer)
    5. EXACT DEDUP  — remove byte-for-byte duplicate documents
    6. NEAR DEDUP   — remove near-duplicate documents (MinHash)
    7. QUALITY FILTER — reject low-quality documents
    8. FINALISE     — write cleaned corpus ready for tokeniser training

Each stage writes to its own intermediate JSONL file so the pipeline can
be restarted from any stage if it fails mid-run.

CLI usage:
    python -m src.data.pipeline \
        --seeds scripts/training/seed_urls.txt \
        --output storage/datasets/ \
        --restrict-domains insightserenity.com \
        --skip-crawl               # If you already have raw HTML
        --start-from quality       # Resume from a specific stage

Programmatic usage:
    from src.data.pipeline import DataPipeline

    pipeline = DataPipeline(output_dir="storage/datasets/")
    pipeline.run(seed_urls=["https://example.com"])
"""

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from src.config.settings import settings
from src.data.crawler.web_crawler import WebCrawler
from src.data.preprocessing.html_extractor import HTMLExtractor
from src.data.preprocessing.text_cleaner import TextCleaner
from src.data.preprocessing.normalizer import TextNormalizer
from src.data.deduplication.exact_dedup import ExactDeduplicator
from src.data.deduplication.minhash import MinHashDeduplicator
from src.data.quality_filter.filters import QualityFilterPipeline
from src.utils.file_io import iter_jsonl, append_jsonl, write_json, ensure_dir, count_lines
from src.utils.logger import get_logger, LogTimer

logger = get_logger(__name__)


def domains_from_seed_urls(seed_urls: List[str]) -> List[str]:
    """Return stable, unique hostnames from a list of seed URLs."""
    domains = []
    seen = set()
    for url in seed_urls:
        domain = urlparse(url).netloc
        if domain and domain not in seen:
            seen.add(domain)
            domains.append(domain)
    return domains


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineStats:
    """
    Tracks document counts and reduction rates across all pipeline stages.
    Written to a JSON summary file at the end of each run.
    """
    crawled:         int = 0
    after_extract:   int = 0
    after_clean:     int = 0
    after_exact_dedup: int = 0
    after_near_dedup:  int = 0
    after_quality:   int = 0
    stage_times:     dict = field(default_factory=dict)

    def reduction_rate(self, before: int, after: int) -> float:
        if before == 0:
            return 0.0
        return round((before - after) / before, 4)

    def summary(self) -> dict:
        return {
            "crawled":                self.crawled,
            "after_extract":          self.after_extract,
            "after_exact_dedup":      self.after_exact_dedup,
            "after_near_dedup":       self.after_near_dedup,
            "after_quality":          self.after_quality,
            "final":                  self.after_quality,
            "exact_dedup_reduction":  self.reduction_rate(self.after_extract, self.after_exact_dedup),
            "near_dedup_reduction":   self.reduction_rate(self.after_exact_dedup, self.after_near_dedup),
            "quality_reduction":      self.reduction_rate(self.after_near_dedup, self.after_quality),
            "total_reduction":        self.reduction_rate(self.crawled, self.after_quality),
            "stage_times":            self.stage_times,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class DataPipeline:
    """
    Full data acquisition and preparation pipeline.

    Manages the sequence of pipeline stages, intermediate file paths,
    and overall statistics. Designed to be idempotent: if an intermediate
    file already exists, that stage is skipped (unless --force is passed).

    Args:
        output_dir:          Root directory for all pipeline output files.
        restrict_domains:    If set, the crawler only follows links to these
                             domains (useful for focused corpora).
        skip_crawl:          Skip the crawl stage (use existing raw file).
        force:               Overwrite existing intermediate files.
    """

    def __init__(
        self,
        output_dir: str = "storage/datasets",
        restrict_domains: Optional[List[str]] = None,
        skip_crawl: bool = False,
        force: bool = False,
    ) -> None:
        self._output_dir        = ensure_dir(output_dir)
        self._restrict_domains  = restrict_domains or []
        self._skip_crawl        = skip_crawl
        self._force             = force

        # Intermediate file paths (one per stage)
        self._raw_html_path     = self._output_dir / "01_raw_html.jsonl"
        self._extracted_path    = self._output_dir / "02_extracted.jsonl"
        self._exact_dedup_path  = self._output_dir / "03_exact_deduped.jsonl"
        self._near_dedup_path   = self._output_dir / "04_near_deduped.jsonl"
        self._final_path        = self._output_dir / "05_corpus.jsonl"
        self._stats_path        = self._output_dir / "pipeline_stats.json"

        self._stats = PipelineStats()

        # Stage components
        self._extractor  = HTMLExtractor()
        self._cleaner    = TextCleaner()
        self._normalizer = TextNormalizer()
        self._quality    = QualityFilterPipeline()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, seed_urls: List[str]) -> PipelineStats:
        """
        Execute the full pipeline synchronously.

        Args:
            seed_urls: List of URLs to seed the crawler with.

        Returns:
            PipelineStats with document counts at each stage.
        """
        logger.info(
            "Pipeline started",
            output_dir=str(self._output_dir),
            seeds=len(seed_urls),
        )

        # Stage 1: Crawl
        if not self._skip_crawl:
            self._stage_crawl(seed_urls)

        # Stage 2: Extract + Clean + Normalise
        self._stage_extract_and_clean()

        # Stage 3: Exact deduplication
        self._stage_exact_dedup()

        # Stage 4: Near-duplicate deduplication
        self._stage_near_dedup()

        # Stage 5: Quality filtering
        self._stage_quality_filter()

        # Write summary
        summary = self._stats.summary()
        write_json(str(self._stats_path), summary)

        logger.info(
            "Pipeline completed",
            **summary,
        )
        return self._stats

    # ── Stages ─────────────────────────────────────────────────────────────────

    def _stage_crawl(self, seed_urls: List[str]) -> None:
        """Stage 1: Crawl seed URLs and save raw HTML to JSONL."""
        if self._raw_html_path.exists() and not self._force:
            self._stats.crawled = count_lines(self._raw_html_path)
            logger.info("Skipping crawl — file exists", path=str(self._raw_html_path))
            return

        self._reset_output_file(self._raw_html_path)

        crawler = WebCrawler(
            output_path=str(self._raw_html_path),
            restrict_domains=self._restrict_domains,
        )

        with LogTimer(logger, "Stage 1: Crawl"):
            asyncio.run(crawler.crawl(seed_urls))

        self._stats.crawled = crawler.pages_crawled
        logger.info("Stage 1 complete", pages_crawled=self._stats.crawled)

    def _stage_extract_and_clean(self) -> None:
        """Stage 2: Extract text from HTML, clean, and normalise."""
        if self._extracted_path.exists() and not self._force:
            self._stats.after_extract = count_lines(self._extracted_path)
            logger.info("Skipping extract — file exists", path=str(self._extracted_path))
            return

        self._reset_output_file(self._extracted_path)

        count = 0
        skipped = 0

        with LogTimer(logger, "Stage 2: Extract + Clean"):
            for record in iter_jsonl(str(self._raw_html_path)):
                html = record.get("html", "")
                url  = record.get("url", "")

                # Extract plain text from HTML
                text = self._extractor.extract(html)
                if not text:
                    skipped += 1
                    continue

                # Clean Unicode noise, URLs, etc.
                text = self._cleaner.clean(text)
                if not text:
                    skipped += 1
                    continue

                # Typographic normalisation
                text = self._normalizer.normalize(text)
                if not text:
                    skipped += 1
                    continue

                # Write cleaned record
                metadata = self._extractor.extract_metadata(html)
                append_jsonl(str(self._extracted_path), {
                    "text":         text,
                    "url":          url,
                    "domain":       record.get("domain", ""),
                    "depth":        record.get("depth", 0),
                    "title":        metadata.get("title", ""),
                    "language":     metadata.get("language", ""),
                    "crawled_at":   record.get("crawled_at", ""),
                })
                count += 1

                if count % 10_000 == 0:
                    logger.info("Extract progress", extracted=count, skipped=skipped)

        self._stats.after_extract = count
        logger.info("Stage 2 complete", extracted=count, skipped=skipped)

    def _stage_exact_dedup(self) -> None:
        """Stage 3: Remove exact duplicate documents."""
        if self._exact_dedup_path.exists() and not self._force:
            self._stats.after_exact_dedup = count_lines(self._exact_dedup_path)
            logger.info("Skipping exact dedup — file exists")
            return

        self._reset_output_file(self._exact_dedup_path)

        with LogTimer(logger, "Stage 3: Exact Dedup"):
            dedup = ExactDeduplicator()
            stats = dedup.deduplicate_file(
                input_path=str(self._extracted_path),
                output_path=str(self._exact_dedup_path),
            )
        self._stats.after_exact_dedup = stats["unique"]
        logger.info("Stage 3 complete", **stats)

    def _stage_near_dedup(self) -> None:
        """Stage 4: Remove near-duplicate documents (MinHash)."""
        if self._near_dedup_path.exists() and not self._force:
            self._stats.after_near_dedup = count_lines(self._near_dedup_path)
            logger.info("Skipping near dedup — file exists")
            return

        self._reset_output_file(self._near_dedup_path)

        with LogTimer(logger, "Stage 4: Near Dedup (MinHash)"):
            dedup = MinHashDeduplicator()
            stats = dedup.deduplicate_file(
                input_path=str(self._exact_dedup_path),
                output_path=str(self._near_dedup_path),
            )
        self._stats.after_near_dedup = stats["unique"]
        logger.info("Stage 4 complete", **stats)

    def _stage_quality_filter(self) -> None:
        """Stage 5: Quality filtering — keep only high-quality documents."""
        if self._final_path.exists() and not self._force:
            self._stats.after_quality = count_lines(self._final_path)
            logger.info("Skipping quality filter — file exists")
            return

        self._reset_output_file(self._final_path)

        with LogTimer(logger, "Stage 5: Quality Filter"):
            stats = self._quality.filter_file(
                input_path=str(self._near_dedup_path),
                output_path=str(self._final_path),
            )
        self._stats.after_quality = stats["passed"]
        logger.info("Stage 5 complete", **stats)

    def _reset_output_file(self, path: Path) -> None:
        """Clear a stage output before recomputing it in forced runs."""
        if self._force and path.exists():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="InsightSerenity Data Pipeline — crawl, clean, deduplicate, filter"
    )
    parser.add_argument(
        "--seeds", required=True,
        help="Path to a text file with one seed URL per line.",
    )
    parser.add_argument(
        "--output", default="storage/datasets/",
        help="Output directory for pipeline files. Default: storage/datasets/",
    )
    parser.add_argument(
        "--restrict-domains", nargs="*",
        help=(
            "Only follow links within these domains (space-separated hostnames). "
            "Defaults to the domains from --seeds."
        ),
    )
    parser.add_argument(
        "--allow-external-domains", action="store_true",
        help="Allow the crawler to follow links outside the seed domains.",
    )
    parser.add_argument(
        "--skip-crawl", action="store_true",
        help="Skip the crawl stage. Assumes 01_raw_html.jsonl already exists.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing intermediate files.",
    )
    args = parser.parse_args()

    # Read seed URLs
    from src.utils.file_io import read_lines
    seed_urls = read_lines(args.seeds)
    if not seed_urls:
        raise ValueError(f"No seed URLs found in {args.seeds}")

    restrict_domains = args.restrict_domains
    if restrict_domains is None and not args.allow_external_domains:
        restrict_domains = domains_from_seed_urls(seed_urls)

    pipeline = DataPipeline(
        output_dir=args.output,
        restrict_domains=restrict_domains,
        skip_crawl=args.skip_crawl,
        force=args.force,
    )
    pipeline.run(seed_urls=seed_urls)


if __name__ == "__main__":
    main()
