import os
import glob
import shutil
import random
import time
import gc
import numpy as np
import torch
import gdown
from pathlib import Path
from getpass import getpass
import json

# LangChain & AI Utils
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

# 디바이스 설정 (GPU 우선 사용)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"현재 사용 중인 디바이스: {DEVICE}")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"모든 랜덤 시드가 {seed}(으)로 고정되었습니다.")

seed_everything(42)

# ==========================================
# 1. 데이터 다운로드 (파일명 강제 지정으로 에러 완벽 해결)
# ==========================================
JSON_SOURCE_DIR = "./data/documents"
os.makedirs(JSON_SOURCE_DIR, exist_ok=True)
print(f"저장 폴더 확인: {JSON_SOURCE_DIR}")

# 구글 드라이브 ID와 저장할 파일명을 1:1로 매핑
file_mapping = [
    ("1XcHKLcIwdJFeSUnwltmfs6pTqkNwFq2l", "rag_data_1.json"),
    ("1DTRFv9MeUu8FBU9MsIpR7NcH5H61JT9x", "rag_data_2.json"),
    ("1bcX1ch-bW1bfuVGbeXt5NwsqYfd_qQA2", "rag_data_3.json"),
    ("14dyKNwNeay4Wjb2ofdbQ2ZbTa8DejVqa", "rag_data_4_DRUG.json"),
    ("12JIGMInjeQZ_RwBBKDkzokU59BRQoQTy", "rag_data_4_DIS.json"),
    ("1Ny7wXs1frZIeZIEPzSjZr9-8ABOmVSwV", "rag_data_4_FAQ.json"),
    ("10yQ3Nv_MuTfdQRG5hA9mPEUomvigENAr", "rag_data_4_INT.json"),
    ("1_SmggkBnogh75PGN7xU_Yjtz69OtP7y3", "rag_data_4_OTH.json"),
    ("1-YlGvAhbU6yRs5uEPAMAUAKjEYLH14bw", "rag_data_4_SYM.json"),
    ("1VtfeMg_SgYpeIItSaYXxJ_acF9Ub9KC1", "rag_data_5_ING.json"),
    ("1wAZZsZ5EnUJqqEQ5e92HUqmlGN_Fd27_", "rag_data_5_INT.json"),
    ("1YDovCucKWG2lX23wNW09Shzf-DyC6Quw", "rag_data_5_OTH.json"),
    ("1FpN7g3m6PhHL_0vyMLGT2gnGPglBVoJ9", "rag_data_5_RAW.json"),
    ("1RYMbDONJa4YezeX-clUipyqu15PUaMJR", "rag_data_5_SAFE.json"),
    ("1EnpEH8eXGl29JC0nPOF-c-_4QNCUmcgs", "rag_data_5_SUPP.json")
]

# 다운로드 실행
for i, (file_id, file_name) in enumerate(file_mapping, 1):
    url = f'https://drive.google.com/uc?id={file_id}'
    out_path = os.path.join(JSON_SOURCE_DIR, file_name)

    # 이미 다운로드 된 파일이 있으면 건너뛰기 (시간 절약)
    if os.path.exists(out_path):
        print(f"[{i}/{len(file_mapping)}] {file_name} 이미 존재함. 다운로드 생략.")
        continue

    print(f"\n[{i}/{len(file_mapping)}] {file_name} 다운로드 중...")
    gdown.download(url, output=out_path, fuzzy=True, quiet=False)

print("\n[다운로드된 폴더 내용물 확인]")
print(os.listdir(JSON_SOURCE_DIR))


# ==========================================
# 2. JSON 파일 로드
# ==========================================
FILE_1 = Path(f"{JSON_SOURCE_DIR}/rag_data_1.json")
FILE_2 = Path(f"{JSON_SOURCE_DIR}/rag_data_2.json")
FILE_3 = Path(f"{JSON_SOURCE_DIR}/rag_data_3.json")
FILE_4 = Path(f"{JSON_SOURCE_DIR}/rag_data_4_DRUG.json")
FILE_5 = Path(f"{JSON_SOURCE_DIR}/rag_data_4_DIS.json")
FILE_6 = Path(f"{JSON_SOURCE_DIR}/rag_data_4_FAQ.json")
FILE_7 = Path(f"{JSON_SOURCE_DIR}/rag_data_4_INT.json")
FILE_8 = Path(f"{JSON_SOURCE_DIR}/rag_data_4_OTH.json")
FILE_9 = Path(f"{JSON_SOURCE_DIR}/rag_data_4_SYM.json")
FILE_10 = Path(f"{JSON_SOURCE_DIR}/rag_data_5_ING.json")
FILE_11 = Path(f"{JSON_SOURCE_DIR}/rag_data_5_INT.json")
FILE_12 = Path(f"{JSON_SOURCE_DIR}/rag_data_5_OTH.json")
FILE_13 = Path(f"{JSON_SOURCE_DIR}/rag_data_5_RAW.json")
FILE_14 = Path(f"{JSON_SOURCE_DIR}/rag_data_5_SAFE.json")
FILE_15 = Path(f"{JSON_SOURCE_DIR}/rag_data_5_SUPP.json")

def drugs_metadata(record, metadata):
    return {"id": record.get("id"), "source": "drugs", "_source_file": record.get("_source_file")}

documents_drugs_1 = JSONLoader(file_path=FILE_1, jq_schema=".[]", content_key=None, metadata_func=drugs_metadata, text_content=False).load()
documents_drugs_2 = JSONLoader(file_path=FILE_2, jq_schema=".[]", content_key=None, metadata_func=drugs_metadata, text_content=False).load()
documents_drugs_3 = JSONLoader(file_path=FILE_3, jq_schema=".[]", content_key=None, metadata_func=drugs_metadata, text_content=False).load()
documents_drugs_4 = JSONLoader(file_path=FILE_4, jq_schema=".[]", content_key=None, metadata_func=drugs_metadata, text_content=False).load()

def disease_metadata(record, metadata):
    return {"id": record.get("id"), "source": "dis", "_source_file": record.get("_source_file")}
documents_disease = JSONLoader(file_path=FILE_5, jq_schema=".[]", content_key=None, metadata_func=disease_metadata, text_content=False).load()

def faq_metadata(record, metadata):
    return {"id": record.get("id"), "source": "faq", "_source_file": record.get("_source_file")}
documents_faq = JSONLoader(file_path=FILE_6, jq_schema=".[]", content_key=None, metadata_func=faq_metadata, text_content=False).load()

def interaction_metadata(record, metadata):
    return {"id": record.get("id"), "source": "interaction", "_source_file": record.get("_source_file")}
documents_interaction = JSONLoader(file_path=FILE_7, jq_schema=".[]", content_key=None, metadata_func=interaction_metadata, text_content=False).load()

def interactions_metadata(record, metadata):
    return {"id": record.get("id"), "source": "interactions", "_source_file": record.get("_source_file") or metadata.get("source")}
documents_interactions = JSONLoader(file_path=FILE_8, jq_schema=".[0].interactions[]", content_key=None, metadata_func=interactions_metadata, text_content=False).load()

def pregnancy_metadata(record, metadata):
    return {"id": record.get("id"), "source": "pregnancy", "_source_file": record.get("_source_file") or metadata.get("source")}
documents_pregnancy = JSONLoader(file_path=FILE_8, jq_schema=".[0].pregnancy_safety[]", content_key=None, metadata_func=pregnancy_metadata, text_content=False).load()

def symptom_metadata(record, metadata):
    return {"id": record.get("id"), "source": "sym", "_source_file": record.get("_source_file") or metadata.get("source")}
documents_symptom = JSONLoader(file_path=FILE_9, jq_schema=".[]", content_key=None, metadata_func=symptom_metadata, text_content=False).load()

def ingredients_metadata(record, metadata):
    return {"id": record.get("id"), "source": "ing", "_source_file": record.get("_source_file") or metadata.get("source")}
documents_ing_raw = JSONLoader(file_path=FILE_10, jq_schema=".", content_key="content", metadata_func=ingredients_metadata, text_content=True).load()
documents_ing = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0).split_documents(documents_ing_raw)

def massive_interaction_metadata(record, metadata):
    return {"id": record.get("id"), "source": "mas", "_source_file": record.get("_source_file") or metadata.get("source")}
documents_massive_int = JSONLoader(file_path=FILE_11, jq_schema=".[]", content_key=None, metadata_func=massive_interaction_metadata, text_content=False).load()

def otc_group_metadata(record, metadata):
    return {"id": record.get("id") or record.get("일반명"), "source": "otc", "_source_file": record.get("_source_file") or metadata.get("source")}
documents_otc_group = JSONLoader(file_path=FILE_12, jq_schema=".[]", content_key=None, metadata_func=otc_group_metadata, text_content=False).load()

def raw_ingredient_metadata(record, metadata):
    return {"id": record.get("id"), "source": "raw", "_source_file": record.get("_source_file") or metadata.get("source")}
documents_raw_ing = JSONLoader(file_path=FILE_13, jq_schema=".[]", content_key=None, metadata_func=raw_ingredient_metadata, text_content=False).load()

def safety_metadata(record, metadata):
    return {"id": record.get("id"), "source": "safety", "_source_file": record.get("_source_file") or metadata.get("source")}
documents_safety = JSONLoader(file_path=FILE_14, jq_schema=".[]", content_key=None, metadata_func=safety_metadata, text_content=False).load()

def supplements_metadata(record, metadata):
    return {"id": record.get("id"), "source": "supp", "_source_file": record.get("_source_file") or metadata.get("source")}
documents_supp = JSONLoader(file_path=FILE_15, jq_schema=".[]", content_key=None, metadata_func=supplements_metadata, text_content=False).load()

all_documents = (
        documents_drugs_1 + documents_drugs_2 + documents_drugs_3 + documents_drugs_4 +
        documents_disease + documents_faq + documents_interaction + documents_interactions +
        documents_pregnancy + documents_symptom + documents_ing + documents_massive_int +
        documents_otc_group + documents_raw_ing + documents_safety + documents_supp
)
print(f"총 문서 수: {len(all_documents)}")


# ==========================================
# 3. 포맷팅 함수 정의 및 실행
# ==========================================
from typing import Any, Dict, List, Union, Optional
from collections import Counter

def _norm(x: Any) -> Optional[str]:
    if x is None: return None
    if isinstance(x, str):
        s = " ".join(x.split()).strip()
        return s if s else None
    if isinstance(x, (int, float, bool)): return str(x)
    try: return json.dumps(x, ensure_ascii=False)
    except Exception: return str(x)

def _bullet(items: Any, indent: str = "  - ") -> str:
    if not isinstance(items, list) or not items: return ""
    cleaned = [v for v in [_norm(v) for v in items] if v]
    return "\n".join(f"{indent}{v}" for v in cleaned) if cleaned else ""

def format_drug_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list):
        if not data: return "[의약품 정보]\n(데이터 없음)"
        data = data[0]
    if not isinstance(data, dict): return f"[의약품 정보]\n(지원하지 않는 형식)"

    def fmt_list(x: Any) -> Optional[str]:
        if not isinstance(x, list): return _norm(x)
        items = [v for v in [_norm(v) for v in x] if v]
        return ", ".join(items) if items else None

    lines: List[str] = []
    def add(label: str, value: Any, *, is_list: bool = False):
        v = fmt_list(value) if is_list else _norm(value)
        if v: lines.append(f"{label}: {v}")

    header = "[의약품 정보]"
    add("제품명", data.get("제품명"))
    add("성분명", data.get("성분명"))
    add("성분명_영문", data.get("성분명_영문"))
    add("분류", data.get("분류"))
    add("제형", data.get("제형"))
    add("제조사", data.get("제조사"))
    add("효능효과", data.get("효능효과"))
    add("용법용량", data.get("용법용량"))

    preg_grade, preg_desc = data.get("임신등급"), data.get("임신등급_설명")
    if _norm(preg_grade) or _norm(preg_desc):
        lines.append("임신등급:")
        if _norm(preg_grade): lines.append(f"  - 등급: {_norm(preg_grade)}")
        if _norm(preg_desc): lines.append(f"  - 설명: {_norm(preg_desc)}")

    add("수유안전성", data.get("수유안전성"))
    add("주의사항", data.get("주의사항"))
    add("보관", data.get("보관"))
    add("일반명", data.get("일반명"))
    add("다른표현", data.get("다른표현"), is_list=True)
    add("source_file", data.get("_source_file"))

    return f"{header}\n" + ("\n".join(lines) if lines else "(표시할 정보 없음)")

def format_disease_drug_guide_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list): data = data[0] if data else {}
    content = data.get("content") or {}

    cond_kr = _norm(data.get("condition_kr")) or _norm(content.get("disease_korean"))
    cond_en = _norm(data.get("condition_en")) or _norm(content.get("disease_english"))

    header_lines = ["[질환-약물 가이드]"]
    if cond_kr or cond_en: header_lines.append(f"질환: {cond_kr or ''}" + (f" ({cond_en})" if cond_en else ""))

    drug_classes = content.get("drug_classes")
    drug_section = ""
    if isinstance(drug_classes, list):
        blocks = []
        for i, dc in enumerate(drug_classes, start=1):
            if not isinstance(dc, dict): continue
            cls = _norm(dc.get("class"))
            fl_txt = "✅ 1차 치료" if dc.get("first_line") is True else ("➖ 1차 치료 아님" if dc.get("first_line") is False else "")
            title = f"{i}. {cls or '(약물 계열)'}" + (f" — {fl_txt}" if fl_txt else "")

            lines = [title]
            if dc.get("examples"): lines.append("  예시 약물:\n" + _bullet(dc.get("examples")))
            if _norm(dc.get("mechanism")): lines.append(f"  기전: {_norm(dc.get('mechanism'))}")
            if dc.get("special_populations"): lines.append("  특수 환자군:\n" + _bullet(dc.get("special_populations")))
            if dc.get("contraindications"): lines.append("  금기/주의:\n" + _bullet(dc.get("contraindications")))
            blocks.append("\n".join(lines))
        if blocks: drug_section = "약물 계열(Drug classes):\n" + "\n\n".join(blocks)

    combo = _bullet(content.get("combination_therapy"))
    avoid = _bullet(content.get("avoid_combinations"))

    other_sections = []
    if combo: other_sections.append("권장 병용요법:\n" + combo)
    if avoid: other_sections.append("피해야 할 조합:\n" + avoid)

    parts = ["\n".join(header_lines), drug_section, "\n\n".join(other_sections)]
    return "\n\n".join([p for p in parts if p.strip()]).strip()

def format_faq_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list): data = data[0] if data else {}

    lines = ["[FAQ]"]
    if _norm(data.get("id")): lines.append(f"ID: {_norm(data.get('id'))}")
    if _norm(data.get("카테고리")): lines.append(f"카테고리: {_norm(data.get('카테고리'))}")
    if _norm(data.get("질문")): lines.append(f"\nQ. {_norm(data.get('질문'))}")
    if _norm(data.get("답변")): lines.append(f"A. {_norm(data.get('답변'))}")
    if _norm(data.get("관련약물")): lines.append(f"\n- 관련약물: {_norm(data.get('관련약물'))}")
    return "\n".join(lines).strip()

def format_interaction_list_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list): data = data[0] if data else {}

    lines = ["[약물 상호작용]"]
    if _norm(data.get("category")): lines.append(f"카테고리: {_norm(data.get('category'))}")
    if data.get("substances"): lines.append("기준 물질:\n" + _bullet(data.get("substances")))

    inters = data.get("interactions_list") or []
    if inters:
        lines.append(f"\n상호작용 상세 ({len(inters)}건):")
        for i, it in enumerate(inters, 1):
            if isinstance(it, dict):
                lines.append(f"{i}. {_norm(it.get('interacting_with'))} — [{_norm(it.get('severity'))}]")
                if _norm(it.get("effect")): lines.append(f"  영향: {_norm(it.get('effect'))}")
                if _norm(it.get("recommendation")): lines.append(f"  권고: {_norm(it.get('recommendation'))}")
    return "\n".join(lines).strip()

def format_interactions_pregnancy_bundle_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list): data = data[0] if data else {}

    lines = ["[상호작용 + 임신/수유 안전성]"]
    inters = data.get("interactions") or []
    if inters:
        lines.append("\n상호작용 샘플:")
        for i, it in enumerate(inters[:5], 1):
            if isinstance(it, dict):
                lines.append(f"{i}. {_norm(it.get('약물1'))} ↔ {_norm(it.get('물질2'))} — 위험도[{_norm(it.get('위험도'))}]")
                if _norm(it.get("권고사항")): lines.append(f"  권고: {_norm(it.get('권고사항'))}")

    preg = data.get("pregnancy_safety") or []
    if preg:
        lines.append("\n임신/수유 안전성 샘플:")
        for i, p in enumerate(preg[:5], 1):
            if isinstance(p, dict):
                lines.append(f"{i}. {_norm(p.get('약물명'))} — FDA[{_norm(p.get('FDA등급'))}]")
                if _norm(p.get("등급설명")): lines.append(f"  설명: {_norm(p.get('등급설명'))}")
    return "\n".join(lines).strip()

def format_symptom_otc_guide_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list): data = data[0] if data else {}
    content = data.get("content") or {}

    lines = ["[증상 자가관리(OTC) 가이드]"]
    if _norm(content.get("symptom")): lines.append(f"증상: {_norm(content.get('symptom'))}")

    otc = content.get("otc_options") or []
    if otc:
        lines.append("\nOTC 선택지:")
        for i, opt in enumerate(otc, 1):
            if isinstance(opt, dict):
                lines.append(f"{i}. {_norm(opt.get('drug'))} (용량: {_norm(opt.get('dose'))})")

    if content.get("when_to_see_doctor"):
        lines.append("\n병원 진료가 필요한 경우:\n" + _bullet(content.get("when_to_see_doctor")))
    return "\n".join(lines).strip()

def format_interaction_summary_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list): data = data[0] if data else {}

    lines = ["[약물 상호작용(요약)]"]
    if _norm(data.get("severity")): lines.append(f"중증도: {_norm(data.get('severity'))}")
    if _norm(data.get("effect")): lines.append(f"영향: {_norm(data.get('effect'))}")
    if _norm(data.get("management")): lines.append(f"관리: {_norm(data.get('management'))}")
    return "\n".join(lines).strip()

def format_generic_drug_card_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list): data = data[0] if data else {}

    lines = ["[일반명 카드]"]
    if _norm(data.get("일반명")): lines.append(f"일반명: {_norm(data.get('일반명'))}")
    if _norm(data.get("효능효과")): lines.append(f"효능효과: {_norm(data.get('효능효과'))}")
    if _norm(data.get("용법용량")): lines.append(f"용법용량: {_norm(data.get('용법용량'))}")
    if _norm(data.get("주의사항")): lines.append(f"주의사항: {_norm(data.get('주의사항'))}")
    if _norm(data.get("부작용")): lines.append(f"부작용: {_norm(data.get('부작용'))}")
    return "\n".join(lines).strip()

def format_supplement_raw_material_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list): data = data[0] if data else {}
    names = data.get("names") or {}

    lines = ["[건강기능식품 원료]"]
    if _norm(names.get("raw_material_name")): lines.append(f"원료명: {_norm(names.get('raw_material_name'))}")
    if data.get("effects"): lines.append("\n기능/효과:\n" + _bullet(data.get("effects")))
    if data.get("side_effects"): lines.append("\n부작용:\n" + _bullet(data.get("side_effects")))
    if _norm(data.get("daily_intake")): lines.append(f"\n일일 섭취량: {_norm(data.get('daily_intake'))}")
    return "\n".join(lines).strip()

def format_population_safety_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list): data = data[0] if data else {}

    lines = ["[안전성 가이드(인구집단)]"]
    if _norm(data.get("drug_name_kr")): lines.append(f"약물: {_norm(data.get('drug_name_kr'))}")
    if _norm(data.get("risk_group")): lines.append(f"대상군: {_norm(data.get('risk_group'))}")
    if _norm(data.get("risk_level")): lines.append(f"위험 수준: {_norm(data.get('risk_level'))}")
    if _norm(data.get("management")): lines.append(f"관리: {_norm(data.get('management'))}")
    return "\n".join(lines).strip()

def format_supplement_ingredient_guide_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list): data = data[0] if data else {}
    content = data.get("content") or {}

    lines = ["[건기식/허브 가이드]"]
    if _norm(content.get("name")): lines.append(f"성분/허브: {_norm(content.get('name'))}")
    if content.get("common_uses"): lines.append("\n주요 사용 목적:\n" + _bullet(content.get("common_uses")))
    if content.get("cautions"): lines.append("\n주의사항:\n" + _bullet(content.get("cautions")))
    return "\n".join(lines).strip()

# 전체 문서 포맷팅 적용
print("텍스트 포맷팅 변환 중...")
for doc in all_documents:
    source = doc.metadata.get("source")
    if source == "drugs": doc.page_content = format_drug_text(doc.page_content)
    elif source == "dis": doc.page_content = format_disease_drug_guide_text(doc.page_content)
    elif source == "faq": doc.page_content = format_faq_text(doc.page_content)
    elif source == "interaction": doc.page_content = format_interaction_list_text(doc.page_content)
    elif source == "interactions": doc.page_content = format_interactions_pregnancy_bundle_text(doc.page_content)
    elif source == "pregnancy": doc.page_content = format_faq_text(doc.page_content)
    elif source == "sym": doc.page_content = format_symptom_otc_guide_text(doc.page_content)
    elif source == "mas": doc.page_content = format_interaction_summary_text(doc.page_content)
    elif source == "otc": doc.page_content = format_generic_drug_card_text(doc.page_content)
    elif source == "raw": doc.page_content = format_supplement_raw_material_text(doc.page_content)
    elif source == "safety": doc.page_content = format_population_safety_text(doc.page_content)
    elif source == "supp": doc.page_content = format_supplement_ingredient_guide_text(doc.page_content)


# ==========================================
# 4. 임베딩 및 Chroma DB 생성 (고정 경로 덮어쓰기)
# ==========================================
print("🧬 임베딩 모델 로드 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="dragonkue/BGE-m3-ko",
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

SAVE_PATH = Path("./data/chroma_db/latest_index")

if SAVE_PATH.exists():
    print(f"🗑️ 기존 DB 폴더 삭제 중... ({SAVE_PATH})")
    shutil.rmtree(SAVE_PATH)

print(f"💾 Vector DB 생성 및 저장 시작... (약 5~10분 소요)")
doc_vs = Chroma.from_documents(
    documents=all_documents,
    embedding=embeddings,
    persist_directory=str(SAVE_PATH),
)

print(f"✅ 로컬 Chroma DB 생성 완벽하게 끝났습니다! 저장 위치: {SAVE_PATH}")