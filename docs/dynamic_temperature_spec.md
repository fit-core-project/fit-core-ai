# P1-5 readiness_level 기반 Dynamic Temperature 스펙

**버전:** 1.1  
**작성일:** 2026-05-27  
**단계:** P1-5 (스펙 설계 — production 변경 없음)  
**상태:** 매핑 테이블 옵션 확정 대기 → Codex 구현 대기

---

## 1. 설계 요약

AI 루틴 생성기에서 LLM temperature를 `readiness_level`에 따라 동적으로 조정한다.  
현재 routine 생성 temperature는 **0** (완전 결정론적)이다. `readiness_level`이 높을수록 약간 더 다양한 루틴 조합을 허용하고, 낮을수록 보수적·안정적 출력을 유지한다.

**핵심 원칙:**
- `ENABLE_DYNAMIC_TEMPERATURE=false`가 기본값 — 배포 즉시 production 동작 변화 없음
- 순수 함수 `resolve_generation_temperature()`로 로직 분리 → 완전 테스트 가능
- repair/fallback 경로는 항상 temperature=0 유지
- 기존 테스트 210개 전부 통과 유지
- Offline Evaluation Harness, Rule Critic과 독립적

---

## 2. 현재 Temperature 설정 위치

### 2-1. 하드코딩 위치

| 파일 | 라인 | 코드 | 비고 |
|------|------|------|------|
| `engines/llm_router.py` | 71 | `def get_llm(engine_type: str, temperature: float = 0):` | 파라미터 기본값 |
| `engines/llm_router.py` | 99–102 | `ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)` | Gemini 클라이언트 생성 |
| `engines/routine_pipeline.py` | 231 | `llm = get_llm("routine")` | temperature 미전달 → 기본값 0 |
| `engines/supplement_engine.py` | 33–34 | `get_llm("supplement", temperature=0.1)` / `temperature=0.0` | routine과 무관 |
| `engines/nlp_engine.py` | 36 | `get_llm("nlp", temperature=0.1)` | routine과 무관 |

### 2-2. 핵심 결론

- **routine 생성 temperature 현재값: `0`** (완전 결정론적)
- temperature는 `get_llm()` 생성 시 1회 주입되며 Gemini 클라이언트에 그대로 전달됨
- repair/retry 시에도 동일 LLM 인스턴스를 재사용하므로 same temperature가 적용됨

---

## 3. readiness_level 정의

### 3-1. 스키마 정의

**파일:** `engines/schemas.py` 라인 35

```python
readiness_level: Optional[str] = "normal"
```

| 항목 | 내용 |
|------|------|
| 타입 | `Optional[str]` |
| 기본값 | `"normal"` |
| 가능한 값 | `"low"`, `"normal"`, `"high"` |
| None 처리 | `readiness_level or "normal"` 패턴으로 전체 코드에서 일관 처리 |

### 3-2. 사용 패턴 (기존 코드)

```python
# prescription/adjustments.py:12
readiness = (readiness_level or "normal").strip().lower()
# → "low": COMPOUND 운동 회피
# → "high": 볼륨 인플레이션 없음

# routine_pipeline.py:220
"readiness_level": req.readiness_level or "normal"

# rule_critic.py:77
readiness_level=req.readiness_level or "normal"
```

### 3-3. 결론

readiness_level은 1~5 숫자가 아닌 **3단계 문자열 enum** (`"low"` / `"normal"` / `"high"`)이다.  
스펙 초안의 숫자 매핑은 이 구조에 맞게 3단계로 재설계한다.

---

## 4. Dynamic Temperature Mapping

### 4-1. ⚠️ 매핑 옵션 — 구현 전 확정 필요

현재 routine temperature는 **0** (완전 결정론적)이다.  
원칙 "보통 readiness: current default 유지"에 따라 "normal"을 어떻게 처리할지 3가지 옵션이 있다.  
**Codex 구현 전 하나를 선택해야 한다.**

| 옵션 | `"low"` | `"normal"` | `"high"` | 특징 |
|------|---------|-----------|---------|------|
| **A (원칙 엄수)** | `0.0` | `0.0` | `0.2` | normal은 현재값 유지. high만 분기. |
| **B (절충)** | `0.0` | `0.1` | `0.2` | normal 소폭 상향. low는 deterministic 유지. |
| **C (전 구간 상향)** | `0.1` | `0.2` | `0.3` | flag ON 시 normal day도 stochastic으로 변함. |

**기본 추천: 옵션 A**
- "production behavior를 급격히 바꾸지 말 것" 원칙에 가장 충실
- flag ON이어도 normal 케이스는 기존 동작 유지
- high만 다양성 확대 → 변화 범위 최소

**옵션 C 주의사항:**  
flag ON 시 모든 정상 케이스(normal)가 temperature 0→0.2로 변함 — 예상치 못한 production 변화 가능성 있음.

| 항목 | 값 |
|------|-----|
| missing / invalid readiness | `default_temperature` (항상 0.0) |
| flag off | 항상 `default_temperature=0.0` |
| 상한 | 0.3 이하 권장 |

### 4-2. 결정 후 _READINESS_TEMPERATURE_MAP 반영

선택된 옵션의 값을 `engines/temperature_policy.py`의 `_READINESS_TEMPERATURE_MAP`에 반영.  
이 매핑 외 코드 구조는 옵션과 무관하게 동일하다.

---

## 5. Feature Flag / Config 정책

### 5-1. 환경변수

**파일:** `fit-core-ai/.env` 및 `.env.example`에 추가

```dotenv
# Dynamic Temperature (기본값 false — production 동작 변경 없음)
ENABLE_DYNAMIC_TEMPERATURE=false
```

### 5-2. 로드 위치

**파일:** `engines/llm_router.py` 또는 신규 `engines/temperature_policy.py`

```python
import os

ENABLE_DYNAMIC_TEMPERATURE: bool = (
    os.getenv("ENABLE_DYNAMIC_TEMPERATURE", "false").lower() == "true"
)
```

### 5-3. 정책

| 상태 | 동작 |
|------|------|
| `ENABLE_DYNAMIC_TEMPERATURE=false` (기본) | 항상 `default_temperature=0.0` 반환 |
| `ENABLE_DYNAMIC_TEMPERATURE=true` | readiness_level → temperature 매핑 적용 |
| 환경변수 누락 | `false`로 처리 (안전 기본값) |

> **주의:** `_DYNAMIC_ENABLED`는 **모듈 import 시점에 캐싱**된다.  
> 런타임에 환경변수를 변경해도 재시작 없이는 반영되지 않는다.  
> 테스트에서는 `_override_enabled` 파라미터로 우회한다.

---

## 6. Repair / Fallback Path 정책

### 6-1. repair/fallback 경로 분류

| 경로 | 파일 | 라인 | 현재 동작 | Temperature 정책 |
|------|------|------|----------|-----------------|
| JSON bracket repair | `llm_parser.py` | 15–30 | LLM 호출 없음 | 무관 |
| OutputFixingParser retry | `llm_parser.py` | 92–102 | `llm` 파라미터로 주입 | **항상 temperature=0** |
| Pipeline 비구조화 재호출 | `routine_pipeline.py` | 246 | `prompt \| llm` — 같은 인스턴스 | **repair_llm으로 교체** |
| Pipeline normalize 재호출 | `routine_pipeline.py` | 248 | `normalize_llm_response(..., llm=llm)` — 같은 인스턴스 | **repair_llm으로 교체** |
| `_fallback()` deterministic path | `routine_pipeline.py` | 50–69 | LLM 호출 없음 | 무관 |

### 6-2. 실제 코드 구조 (직접 확인)

```python
# routine_pipeline.py:240-264 현재 코드
except (OutputParserException, ValidationError) as e:
    try:
        raw_text = getattr(e, "llm_output", None)
        if raw_text is None:
            raw_resp = (prompt | llm).invoke(invoke_kwargs)   # ← 동일 llm
            raw_text = raw_resp.content ...
        normalized = normalize_llm_response(raw_text, llm=llm)  # ← 동일 llm
        ...
```

`llm` 인스턴스가 동적 temperature를 갖고 있으면 repair도 동일 temperature로 실행된다.  
이를 막기 위해 **repair_llm을 별도 생성**해야 한다.

### 6-3. 구현 원칙

```
1차 LLM 호출: resolve_generation_temperature(readiness_level) → 동적 temperature
repair LLM 호출: temperature=0 고정 → get_llm("routine", temperature=0)
fallback 경로: LLM 없음 → temperature 무관
```

**변경 대상:**  
`routine_pipeline.py:240-261` except 블록 진입 시점에 `repair_llm = get_llm("routine", temperature=0)` 생성 후  
라인 246의 `llm` → `repair_llm`, 라인 248의 `llm=llm` → `llm=repair_llm`으로 교체.

---

## 7. 파일 구조 제안

### 7-1. 신규 파일

**`engines/temperature_policy.py`** — 순수 함수 모듈

```python
"""
readiness_level 기반 LLM temperature 결정 정책.
ENABLE_DYNAMIC_TEMPERATURE=true일 때만 매핑 적용.
"""
import os
from typing import Optional

_DYNAMIC_ENABLED: bool = (
    os.getenv("ENABLE_DYNAMIC_TEMPERATURE", "false").lower() == "true"
)

_READINESS_TEMPERATURE_MAP: dict[str, float] = {
    "low": 0.1,
    "normal": 0.2,
    "high": 0.3,
}

_DEFAULT_TEMPERATURE: float = 0.0


def resolve_generation_temperature(
    readiness_level: Optional[str],
    default_temperature: float = _DEFAULT_TEMPERATURE,
    *,
    _override_enabled: Optional[bool] = None,  # 테스트 전용
) -> float:
    """
    readiness_level에 따른 generation temperature 반환.

    - ENABLE_DYNAMIC_TEMPERATURE=false (기본): 항상 default_temperature
    - ENABLE_DYNAMIC_TEMPERATURE=true: readiness_level 매핑 적용
    - invalid/missing readiness_level: default_temperature로 폴백
    """
    enabled = _DYNAMIC_ENABLED if _override_enabled is None else _override_enabled
    if not enabled:
        return default_temperature
    normalized = (readiness_level or "normal").strip().lower()
    return _READINESS_TEMPERATURE_MAP.get(normalized, default_temperature)
```

### 7-2. 수정 파일

#### `engines/routine_pipeline.py` — 라인 231 수정

**변경 전:**
```python
llm = get_llm("routine")
```

**변경 후:**
```python
from engines.temperature_policy import resolve_generation_temperature

generation_temp = resolve_generation_temperature(req.readiness_level)
llm = get_llm("routine", temperature=generation_temp)
```

#### repair LLM 인스턴스 분리 (라인 240–261 영역)

**변경 전:**
```python
# OutputFixingParser가 동일 llm 인스턴스 재사용
```

**변경 후:**
```python
repair_llm = get_llm("routine", temperature=0)
# OutputFixingParser(llm=repair_llm, ...) 로 전달
```

> **Note:** `normalize_llm_response()` 시그니처가 `llm` 파라미터를 이미 받으므로, 호출 시 `repair_llm`을 명시적으로 전달한다.

#### `fit-core-ai/.env`

```dotenv
ENABLE_DYNAMIC_TEMPERATURE=false
```

### 7-3. 신규 테스트 파일

**`tests/test_temperature_policy.py`**

---

## 8. 테스트 계획

### 8-1. `tests/test_temperature_policy.py` — 신규 (11개 케이스)

| # | 테스트명 | 입력 | 기대값 |
|---|---------|------|--------|
| T01 | `test_low_readiness_returns_low_temp` | `"low"`, enabled=True | `0.1` |
| T02 | `test_normal_readiness_returns_mid_temp` | `"normal"`, enabled=True | `0.2` |
| T03 | `test_high_readiness_returns_high_temp` | `"high"`, enabled=True | `0.3` |
| T04 | `test_missing_readiness_returns_default` | `None`, enabled=True | `0.0` (default) |
| T05 | `test_invalid_readiness_returns_default` | `"extreme"`, enabled=True | `0.0` (default) |
| T06 | `test_empty_string_readiness_returns_normal_map` | `""`, enabled=True | `0.2` ("" → "normal") |
| T07 | `test_flag_off_ignores_readiness_low` | `"low"`, enabled=False | `0.0` |
| T08 | `test_flag_off_ignores_readiness_high` | `"high"`, enabled=False | `0.0` |
| T09 | `test_flag_off_ignores_readiness_normal` | `"normal"`, enabled=False | `0.0` |
| T10 | `test_custom_default_temperature_respected` | `None`, enabled=True, default=0.5 | `0.5` |
| T11 | `test_case_insensitive_readiness` | `"LOW"`, enabled=True | `0.1` |

### 8-2. 기존 테스트 영향 분석

| 테스트 파일 | 영향 여부 | 이유 |
|------------|----------|------|
| `test_contract.py` | 없음 | readiness_level 파싱만 검증, temperature 무관 |
| `test_fallback.py` | 없음 | deterministic fallback 경로, LLM 호출 없음 |
| `test_prompt_regression.py` | 없음 | 프롬프트 빌딩만 검증 |
| `test_pipeline.py` | 없음 | LLM mock 사용, temperature 값 검증 없음 |
| 나머지 207개 | 없음 | flag 기본값 false → production 동작 동일 |

**기존 210개 테스트 전부 통과 유지 보장.**

### 8-3. repair path 온전성 확인 (선택)

```python
def test_repair_llm_uses_zero_temperature(monkeypatch):
    """
    schema repair 경로에서 get_llm("routine", temperature=0)이
    호출되는지 확인.
    """
    captured = []
    original_get_llm = get_llm

    def mock_get_llm(engine, temperature=0.0):
        captured.append(temperature)
        return original_get_llm(engine, temperature)

    monkeypatch.setattr("engines.routine_pipeline.get_llm", mock_get_llm)
    # schema parse 실패를 유발하는 픽스처로 pipeline 호출
    # captured[-1] == 0 확인 (repair 시 temperature=0)
```

---

## 9. 리스크

| 리스크 | 심각도 | 완화 방법 |
|--------|--------|----------|
| `ENABLE_DYNAMIC_TEMPERATURE=true` 미설정으로 실수 배포 | 낮음 | 기본값 `false`이므로 동작 변화 없음 |
| temperature 상승이 실제 LLM 품질을 보장하지 않음 | 중간 | 스펙 명시: 1차 구현에서 출력 품질 보장 불포함. Offline Harness 단계에서 별도 측정 |
| repair LLM 인스턴스 분리 누락 시 repair가 고온으로 실행됨 | 중간 | `repair_llm = get_llm("routine", temperature=0)` 명시적 분리 필수 |
| 환경변수 타입 오류 (`"True"`, `"TRUE"` 등) | 낮음 | `.lower() == "true"` 패턴으로 처리 |
| readiness_level에 숫자 문자열 `"3"` 입력 가능성 | 낮음 | `_READINESS_TEMPERATURE_MAP.get(normalized, default)` → default 폴백 처리 |
| Offline Evaluation Harness LLM 호출 없음 | 없음 | 설계상 영향 없음 |
| Rule Critic 독립성 | 없음 | temperature와 Rule Critic은 완전 독립 |

---

## 10. Codex 구현용 최종 지시문

```
작업명: P1-5 readiness_level 기반 Dynamic Temperature 구현

전제:
- fit-core-ai/ 디렉터리 기준
- 기존 pytest 210개 모두 통과 유지
- production default 동작 변경 없음 (feature flag off가 기본)

--- Step 1: temperature_policy.py 생성 ---

파일 생성: engines/temperature_policy.py

내용:
- 환경변수 ENABLE_DYNAMIC_TEMPERATURE (기본값 "false") 읽기
- _READINESS_TEMPERATURE_MAP = {"low": 0.1, "normal": 0.2, "high": 0.3}
- _DEFAULT_TEMPERATURE = 0.0
- 함수 resolve_generation_temperature(
      readiness_level: Optional[str],
      default_temperature: float = _DEFAULT_TEMPERATURE,
      *,
      _override_enabled: Optional[bool] = None,
  ) -> float
  - _override_enabled가 None이면 환경변수값 사용 (테스트용 주입 지원)
  - enabled=False → default_temperature 반환
  - enabled=True → (readiness_level or "normal").strip().lower() 정규화 후 매핑
  - 매핑 미스 → default_temperature 반환

--- Step 2: routine_pipeline.py 수정 ---

파일: engines/routine_pipeline.py

변경 위치: 라인 231 근처 `llm = get_llm("routine")` 코드

변경 내용:
  1. 파일 상단에 import 추가:
     from engines.temperature_policy import resolve_generation_temperature

  2. llm 생성 코드를 다음으로 교체:
     generation_temp = resolve_generation_temperature(req.readiness_level)
     llm = get_llm("routine", temperature=generation_temp)

  3. schema repair / OutputFixingParser 호출 영역 (라인 240-264):
     except (OutputParserException, ValidationError) 블록 맨 위에:
       repair_llm = get_llm("routine", temperature=0)
     그 뒤 블록 내부에서:
       - 라인 246: `(prompt | llm).invoke(...)` → `(prompt | repair_llm).invoke(...)`
       - 라인 248: `normalize_llm_response(raw_text, llm=llm)` → `normalize_llm_response(raw_text, llm=repair_llm)`

--- Step 3: .env 수정 ---

파일: fit-core-ai/.env

기존 내용 끝에 추가:
ENABLE_DYNAMIC_TEMPERATURE=false

--- Step 4: 테스트 파일 생성 ---

파일 생성: tests/test_temperature_policy.py

테스트 케이스 (총 11개, 전부 _override_enabled 파라미터 사용 — 환경변수 의존 없음):

T01: resolve_generation_temperature("low", _override_enabled=True) == 0.1
T02: resolve_generation_temperature("normal", _override_enabled=True) == 0.2
T03: resolve_generation_temperature("high", _override_enabled=True) == 0.3
T04: resolve_generation_temperature(None, _override_enabled=True) == 0.0
T05: resolve_generation_temperature("extreme", _override_enabled=True) == 0.0
T06: resolve_generation_temperature("", _override_enabled=True) == 0.2  # "" → "normal"
T07: resolve_generation_temperature("low", _override_enabled=False) == 0.0
T08: resolve_generation_temperature("high", _override_enabled=False) == 0.0
T09: resolve_generation_temperature("normal", _override_enabled=False) == 0.0
T10: resolve_generation_temperature(None, default_temperature=0.5, _override_enabled=True) == 0.5
T11: resolve_generation_temperature("LOW", _override_enabled=True) == 0.1  # case-insensitive

--- 검증 ---

1. pytest tests/test_temperature_policy.py -v → 11개 전부 pass
2. pytest --tb=short → 기존 210개 포함 전체 pass
3. ENABLE_DYNAMIC_TEMPERATURE=false 상태에서 routine_pipeline 동작이
   기존과 동일한지 확인 (get_llm("routine", temperature=0.0) 호출 여부)

--- 비범위 (건드리지 말 것) ---

- prompt_builder.py 변경 없음
- candidate_ranker.py 변경 없음
- rule_critic.py 변경 없음
- feedback API 변경 없음
- schemas.py readiness_level 정의 변경 없음
- candidate pool default 변경 없음
- Offline Evaluation Harness 변경 없음
```

---

## 부록: 관련 파일 위치 요약

| 파일 | 역할 |
|------|------|
| `engines/llm_router.py:71` | `get_llm()` 파라미터 기본값 temperature=0 |
| `engines/llm_router.py:99` | `ChatGoogleGenerativeAI(temperature=temperature)` |
| `engines/routine_pipeline.py:231` | `llm = get_llm("routine")` ← 수정 대상 |
| `engines/schemas.py:35` | `readiness_level: Optional[str] = "normal"` |
| `engines/temperature_policy.py` | **신규** — 순수 함수 매핑 로직 |
| `tests/test_temperature_policy.py` | **신규** — 11개 유닛 테스트 |
| `fit-core-ai/.env` | `ENABLE_DYNAMIC_TEMPERATURE=false` 추가 |
