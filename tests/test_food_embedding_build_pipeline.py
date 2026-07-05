from scripts.build_food_db import build_documents


def _row(**overrides):
    row = {
        "food_id": "F0001",
        "name": "달걀 삶은것",
        "embed_text": "달걀 삶은것",
        "rep_name": "달걀",
        "basis_unit": "g",
        "major_category": "축산물",
        "data_type": "원재료성 식품",
        "basis_value": "100",
        "kcal": "143",
        "carb_g": "1.1",
        "protein_g": "12.6",
        "fat_g": "9.5",
    }
    row.update(overrides)
    return row


def test_build_documents_uses_enriched_embedding_text():
    texts, metadatas, ids, skipped = build_documents([_row()])

    assert skipped == 0
    assert ids == ["F0001"]
    assert len(texts) == 1
    assert "달걀 삶은것" in texts[0]
    assert "삶은 계란" in texts[0]
    assert "삶은계란" in texts[0]
    assert "계란 삶은것" in texts[0]

    assert metadatas[0]["name"] == "달걀 삶은것"
    assert metadatas[0]["rep_name"] == "달걀"
    assert metadatas[0]["data_type"] == "원재료성 식품"


def test_build_documents_keeps_aliases_out_of_metadata():
    texts, metadatas, _, _ = build_documents([_row()])

    assert "삶은 계란" in texts[0]
    metadata_text = " ".join(str(value) for value in metadatas[0].values())
    assert "삶은 계란" not in metadata_text
    assert "삶은계란" not in metadata_text
    assert "계란 삶은것" not in metadata_text


def test_build_documents_does_not_leak_protected_compound_aliases():
    texts, metadatas, ids, skipped = build_documents(
        [
            _row(
                food_id="F0002",
                name="닭고기 가슴(껍질 제거) 생것",
                embed_text="닭고기 가슴(껍질 제거) 생것",
                rep_name="닭고기 가슴",
            )
        ]
    )

    assert skipped == 0
    assert ids == ["F0002"]
    assert "닭가슴살 생것" in texts[0]
    assert "닭 가슴살 생것" in texts[0]
    assert "샐러드 닭가슴살" not in texts[0]
    assert "닭가슴살 샐러드" not in texts[0]
    assert "샌드위치 닭가슴살" not in texts[0]
    assert metadatas[0]["name"] == "닭고기 가슴(껍질 제거) 생것"


def test_build_documents_unknown_row_does_not_add_broad_aliases():
    texts, metadatas, ids, skipped = build_documents(
        [_row(food_id="F0003", name="고구마", embed_text="고구마", rep_name="")]
    )

    assert skipped == 0
    assert ids == ["F0003"]
    assert texts == ["고구마"]
    assert "군고구마" not in texts[0]
    assert metadatas[0]["name"] == "고구마"
