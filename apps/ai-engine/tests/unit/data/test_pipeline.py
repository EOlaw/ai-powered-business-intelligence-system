import json

from src.data.pipeline import DataPipeline, domains_from_seed_urls
from src.utils.file_io import append_jsonl, count_lines, iter_jsonl


class PassThroughQuality:
    def filter_file(self, input_path: str, output_path: str, text_field: str = "text") -> dict:
        stats = {"total": 0, "passed": 0, "rejected": 0, "reasons": {}}
        for record in iter_jsonl(input_path):
            stats["total"] += 1
            stats["passed"] += 1
            append_jsonl(output_path, record)
        return stats


def test_force_run_resets_append_based_stage_outputs(tmp_path):
    output_dir = tmp_path / "manual_phase1"
    output_dir.mkdir()

    text = " ".join(
        [
            "The clean English training corpus contains useful paragraphs about machine learning,",
            "data processing, tokenization, neural networks, model training, evaluation,",
            "software engineering, reproducibility, and careful dataset construction.",
        ]
        * 3
    )
    html = f"<html><head><title>Manual test</title></head><body><main><p>{text}</p></main></body></html>"
    records = [
        {"url": "https://example.com/a", "domain": "example.com", "depth": 0, "html": html},
        {"url": "https://example.com/duplicate", "domain": "example.com", "depth": 0, "html": html},
    ]
    raw_path = output_dir / "01_raw_html.jsonl"
    raw_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    expected_counts = {
        "02_extracted.jsonl": 2,
        "03_exact_deduped.jsonl": 1,
        "04_near_deduped.jsonl": 1,
        "05_corpus.jsonl": 1,
    }

    for _ in range(2):
        pipeline = DataPipeline(output_dir=str(output_dir), skip_crawl=True, force=True)
        pipeline._quality = PassThroughQuality()

        stats = pipeline.run(seed_urls=["https://example.com"])

        assert stats.after_extract == 2
        assert stats.after_exact_dedup == 1
        assert stats.after_near_dedup == 1
        assert stats.after_quality == 1
        for filename, expected_count in expected_counts.items():
            assert count_lines(output_dir / filename) == expected_count


def test_domains_from_seed_urls_preserves_unique_seed_domains():
    assert domains_from_seed_urls(
        [
            "https://insightserenity.com/",
            "https://insightserenity.com/about.html",
            "https://www.insightserenity.com/blog.html",
        ]
    ) == ["insightserenity.com", "www.insightserenity.com"]
