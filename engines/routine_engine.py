import math
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Request / Response 스키마
# ==========================================
# 🐛 수정 완료: 프론트엔드에서 넘어오는 요청 스키마 복구
class RoutineRequest(BaseModel):
    target_muscles: List[str]
    equipment: List[str]
    time_available_min: int
    pain_areas: List[str]
    # 🌟 프론트엔드에서 넘어올 DOMS 데이터 (예: {"CHEST": 2, "LEG_QUADS": 1}) 추가
    doms_data: Optional[Dict[str, int]] = Field(default_factory=dict)
    goal: str
    user_note: str

class ExercisePlan(BaseModel):
    exercise_name: str = Field(description="운동 종목 이름 (예: 인클라인 덤벨 프레스)")
    target_weight: Optional[int] = Field(description="계산된 목표 중량 (kg). 맨몸이면 null")
    reps: int = Field(description="세트당 반복 횟수")
    sets: int = Field(description="총 세트 수")
    rest_time_sec: int = Field(description="세트 간 휴식 시간 (초)")
    coach_tip: str = Field(description="해당 운동 수행 시 주의사항 1줄")

class RoutineResponse(BaseModel):
    total_estimated_time: int = Field(description="루틴 예상 소요 시간 (분)")
    exercises: List[ExercisePlan]
    overall_feedback: str = Field(description="유저의 user_note와 컨디션에 대한 코치의 답변")

# ==========================================
# 2. 핵심 연산 로직 (파이썬 담당)
# ==========================================
def calculate_1rm(weight: int, reps: int) -> float:
    if reps <= 1: return float(weight)
    return weight * (1 + reps / 30.0)

def apply_periodization(one_rm: float, goal: str) -> tuple[int, int, int]:
    """목표에 따른 중량, 반복수, 휴식시간(초) 산출"""
    if goal == "STRENGTH":
        return round((one_rm * 0.85) / 5) * 5, 5, 180
    elif goal == "HYPERTROPHY":
        return round((one_rm * 0.70) / 5) * 5, 10, 90
    else: # ENDURANCE
        return round((one_rm * 0.50) / 5) * 5, 15, 60

# ==========================================
# 3. AI 파이프라인 (프롬프트 고도화)
# ==========================================
def generate_smart_routine(req: RoutineRequest) -> str:
    # [가상의 유저 DB 데이터]
    user_db_1rm = {
        "BENCH_PRESS": calculate_1rm(85, 5),
        "SQUAT": calculate_1rm(110, 5),
        "DEADLIFT": calculate_1rm(120, 3),
        "OHP": calculate_1rm(55, 5)
    }

    # 1. 훈련 목표에 따른 기준 수치 세팅
    bench_w, bench_r, bench_rest = apply_periodization(user_db_1rm["BENCH_PRESS"], req.goal)
    squat_w, squat_r, squat_rest = apply_periodization(user_db_1rm["SQUAT"], req.goal)

    # 2. 운동 시간 기반 세트 수 제한 연산
    safe_time = max(req.time_available_min, 15)
    max_total_sets = int(safe_time / ((60 + bench_rest) / 60))

    # 3. 🌟 DOMS(근육통) 데이터 파싱 및 해석 문자열 생성
    doms_instructions = ""
    if req.doms_data:
        doms_list = []
        for part, level in req.doms_data.items():
            if level == 1:
                doms_list.append(f"- {part}: 약간 뻐근함 (해당 부위 볼륨을 20% 줄일 것)")
            elif level == 2:
                doms_list.append(f"- {part}: 매우 뻐근함 (해당 부위 운동은 가벼운 자극/고립 위주로 1~2세트만 배정할 것)")
            elif level == 3:
                doms_list.append(f"- {part}: 통증/부상 우려 (해당 부위가 주동근 또는 보조근으로 개입되는 모든 운동 절대 금지)")

        doms_instructions = "\n".join(doms_list)
    else:
        doms_instructions = "현재 근육통이 있는 부위가 없습니다. 정상 볼륨으로 진행하세요."

    # 4. LLM 설정 (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(RoutineResponse)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 운동 생리학 지식이 풍부한 피트니스 AI 코치야.
        시스템이 계산한 [하드 제약 조건]과 [오늘의 근육통 상태]를 절대적으로 지켜서 최적의 루틴을 설계해.
        
        [하드 제약 조건]
        - 타겟 부위: {muscles}
        - 사용 가능 장비: {equipment}
        - 전체 루틴의 총 세트 수 합계는 {max_sets} 세트를 초과해선 안 됨.
        
        [오늘의 근육통(DOMS) 상태 및 조치]
        {doms_instructions}
        
        [DB 연산 중량 가이드]
        만약 루틴에 아래 운동(또는 유사 운동)이 포함된다면, 반드시 이 수치를 적용해:
        - 플랫/인클라인 프레스 계열: {bench_w}kg / {bench_r}회 / 휴식 {bench_rest}초
        - 스쿼트 계열: {squat_w}kg / {squat_r}회 / 휴식 {squat_rest}초
        """),
        ("human", "유저 코멘트: {user_note}")
    ])

    try:
        chain = prompt | structured_llm

        safe_muscles = ", ".join(req.target_muscles) if req.target_muscles else "전신"
        safe_equipment = ", ".join(req.equipment) if req.equipment else "맨몸"
        safe_note = req.user_note if req.user_note else "없음"

        response = chain.invoke({
            "muscles": safe_muscles,
            "equipment": safe_equipment,
            "max_sets": max_total_sets,
            "doms_instructions": doms_instructions,
            "bench_w": bench_w, "bench_r": bench_r, "bench_rest": bench_rest,
            "squat_w": squat_w, "squat_r": squat_r, "squat_rest": squat_rest,
            "user_note": safe_note
        })

        return response.model_dump_json()

    except Exception as e:
        print(f"❌ [AI 엔진 에러]: {str(e)}")
        raise Exception(f"AI 루틴 생성 중 오류가 발생했습니다: {str(e)}")

# ==========================================
# 테스트 실행
# ==========================================
if __name__ == "__main__":
    mock_payload = RoutineRequest(
        target_muscles=["CHEST_UPPER", "SHOULDER_FRONT", "ARM_TRICEPS"],
        equipment=["DUMBBELL", "CABLE", "MACHINE"],
        time_available_min=45,
        pain_areas=["ROTATOR_CUFF"],
        goal="HYPERTROPHY",
        user_note="오늘 바벨 쪽에 사람 너무 많아서 못 가. 그리고 저번 주에 회전근개 살짝 다쳐서 무리 안 가게 부탁해. 다이어트 중이라 땀 좀 내고 싶어."
    )

    print("🤖 연산 및 루틴 생성 중...\n")
    # 🐛 수정 완료: 함수명 불일치 해결 (generate_smart_routine 호출)
    print(generate_smart_routine(mock_payload))