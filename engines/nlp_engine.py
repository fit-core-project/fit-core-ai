from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from engines.log_redaction import sanitize_exception_for_log, summarize_text_for_log
from engines.llm_router import get_llm

# .env 파일에서 GOOGLE_API_KEY를 불러옵니다.
load_dotenv()

# ==========================================
# 1. NLP 파싱을 위한 Pydantic 스키마 정의
# ==========================================
class DietItem(BaseModel):
    food_name: str = Field(description="먹은 음식의 이름 (예: 건면 라면, 닭가슴살)")
    estimated_calories: int = Field(description="해당 음식의 추정 칼로리 (kcal)")
    protein_g: int = Field(description="추정 단백질 (g)")
    carbs_g: int = Field(description="추정 탄수화물 (g)")
    fat_g: int = Field(description="추정 지방 (g)")

class WorkoutItem(BaseModel):
    exercise_name: str = Field(description="운동 종목 이름 (예: 바벨 스쿼트, 벤치프레스)")
    weight_kg: Optional[float] = Field(None, description="수행한 중량 (kg). 맨몸이면 null")
    sets: Optional[int] = Field(None, description="수행한 세트 수. 모르면 1")
    reps: Optional[int] = Field(None, description="수행한 반복 횟수")

class ParsedDailyLog(BaseModel):
    diet_logs: List[DietItem] = Field(default_factory=list, description="추출된 식단 목록")
    workout_logs: List[WorkoutItem] = Field(default_factory=list, description="추출된 운동 목록")
    overall_summary: str = Field(description="입력된 내용에 대한 AI 코치의 짧은 응원이나 피드백 (대화하듯 친근한 반말/존댓말 혼용, 1문장)")

# ==========================================
# 2. 핵심 파싱 파이프라인 (자연어 -> JSON 구조화)
# ==========================================
def parse_natural_language_log(user_text: str) -> str:
    # 온도(temperature)를 0.1로 낮추어 창의성보다는 정확한 정보 추출에 집중하게 합니다.
    llm = get_llm("nlp", temperature=0.1)

    # Pydantic 스키마를 LLM에 강제 적용 (가장 중요한 부분)
    structured_llm = llm.with_structured_output(ParsedDailyLog)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 뛰어난 영양사이자 헬스 트레이너 AI야.
        사용자가 일상적인 언어로 입력한 텍스트에서 '식단'과 '운동' 기록을 분리하고 완벽하게 구조화해.
        
        [행동 지침]
        1. 식단: 음식 이름이 나오면 일반적인 영양소 DB를 기반으로 칼로리와 매크로(단백질, 탄수, 지방)를 추정해서 채워넣어. (예: 닭가슴살 100g -> 110kcal, 단 23g, 탄 0g, 지 1g)
        2. 운동: 종목, 무게, 세트, 횟수를 정확히 추출해. (예: '170kg 3대 쳤다' -> 스쿼트 170kg 3회)
        3. 누락 보정: 사용자가 세트수를 말하지 않았으면 1로, 무게를 말하지 않았으면 맨몸(null)으로 간주해.
        4. 피드백: 'overall_summary'에는 트레이너가 직접 카톡을 보내주는 것처럼 위트있고 친근하게 한 문장으로 코멘트를 작성해.
        """),
        ("human", "오늘 기록할 내용: {text}")
    ])

    try:
        chain = prompt | structured_llm
        print("🤖 [Gemini 파싱 시작] 분석 중인 텍스트:", summarize_text_for_log(user_text))

        # AI가 텍스트를 읽고 ParsedDailyLog 객체 형태로 변환하여 반환
        parsed_result = chain.invoke({"text": user_text})

        # FastAPI(main.py)로 넘겨주기 위해 객체를 JSON 문자열로 직렬화
        return parsed_result.model_dump_json()

    except Exception as e:
        print("❌ [Gemini 파싱 에러 상세]:", sanitize_exception_for_log(e))
        raise Exception("자연어 기록을 AI가 분석하는 중 오류가 발생했습니다.")
