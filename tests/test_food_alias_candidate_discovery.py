from io import StringIO

from scripts.discover_food_alias_candidates import (
    DO_NOT_LINK_PROTECTED_COMPOUND,
    SAFE_SPACING_VARIANT,
    discover_alias_candidates,
    discover_row_alias_candidates,
    emit_report,
    format_markdown_report,
)


def test_discovers_chicken_breast_spacing_variants():
    candidates = discover_row_alias_candidates(
        {
            "name": "닭고기 가슴(껍질 제거) 생것",
            "rep_name": "닭고기",
            "data_type": "원재료성 식품",
            "major_category": "축산물",
        }
    )

    aliases = {candidate.source_alias for candidate in candidates}
    risk_labels = {candidate.source_alias: candidate.risk_label for candidate in candidates}

    assert "닭가슴살 생것" in aliases
    assert "닭 가슴살 생것" in aliases
    assert risk_labels["닭가슴살 생것"] == SAFE_SPACING_VARIANT
    assert risk_labels["닭 가슴살 생것"] == SAFE_SPACING_VARIANT


def test_protected_compound_is_not_safe_alias_for_ingredient():
    candidates = discover_alias_candidates(
        [
            {
                "name": "계란빵",
                "rep_name": "계란빵",
                "data_type": "음식",
                "major_category": "빵류",
            },
            {
                "name": "달걀 생것",
                "rep_name": "달걀",
                "data_type": "원재료성 식품",
                "major_category": "축산물",
            },
        ]
    )

    protected = [candidate for candidate in candidates if candidate.source_alias == "계란빵"]
    assert protected
    assert protected[0].target_name == "계란빵"
    assert protected[0].risk_label == DO_NOT_LINK_PROTECTED_COMPOUND
    assert not any(
        candidate.source_alias == "계란빵"
        and candidate.target_name == "달걀 생것"
        and candidate.risk_label.startswith("SAFE_")
        for candidate in candidates
    )


def test_markdown_report_contains_warning_and_schema():
    candidates = discover_row_alias_candidates(
        {
            "name": "계란빵",
            "rep_name": "계란빵",
            "data_type": "음식",
            "major_category": "빵류",
        }
    )

    report = format_markdown_report(candidates)

    assert "자동 적용 금지" in report
    assert "source_alias" in report
    assert "risk_label" in report
    assert "계란빵" in report
    assert DO_NOT_LINK_PROTECTED_COMPOUND in report


def test_emit_report_writes_only_to_stream_by_default(tmp_path):
    candidates = discover_row_alias_candidates(
        {
            "name": "달걀 삶은것",
            "rep_name": "달걀",
            "data_type": "원재료성 식품",
            "major_category": "축산물",
        }
    )
    stream = StringIO()

    emit_report(candidates, stream, limit=10)

    assert "달걀 삶은것" in stream.getvalue()
    assert list(tmp_path.iterdir()) == []
