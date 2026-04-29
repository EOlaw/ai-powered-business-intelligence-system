from src.data.quality_filter.language_detector import LanguageDetector


def test_detects_english_business_copy_as_allowed():
    text = " ".join(
        [
            "InsightSerenity helps organizations transform data into business intelligence",
            "through data science, software engineering, machine learning, and technical",
            "consulting. The work includes careful discovery, practical strategy, reliable",
            "execution, clear documentation, and ongoing support for teams that need",
            "systems they can understand, operate, and improve over time.",
        ]
        * 3
    )

    detector = LanguageDetector()

    assert detector.detect(text)[0] == "en"
    assert detector.is_allowed(text)
