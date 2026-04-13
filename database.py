from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# MariaDB 연결 URL (pymysql 드라이버 사용)
# 형식: mysql+pymysql://사용자명:비밀번호@호스트:포트/데이터베이스명
DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/fit_core"

# Engine 생성 (연결 풀링 설정 포함)
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600
)

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ORM 모델의 베이스 클래스
Base = declarative_base()

# DB 세션 의존성 주입 (FastAPI 등에서 사용하기 위함)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()