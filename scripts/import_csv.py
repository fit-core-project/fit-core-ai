import pandas as pd
from sqlalchemy import create_engine
import os

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
target_file = "exercise_tier.xlsx"
csv_file_path = os.path.join(BASE_DIR, target_file)

# 2. MariaDB 연결 설정
DATABASE_URL = "mysql+pymysql://root:1234@localhost:3307/fit_core"
engine = create_engine(DATABASE_URL)

def import_data():
    print(f"확인 중인 파일: {csv_file_path}")

    if not os.path.exists(csv_file_path):
        print(f"error: 파일이 없습니다. {target_file} 파일이 scripts 폴더 안에 있는지 확인하세요.")
        return

    try:
        if csv_file_path.endswith('.xlsx'):
            print("엑셀 파일(.xlsx)을 읽는 중...")
            df = pd.read_excel(csv_file_path)
        else:
            print("CSV 파일(.csv)을 읽는 중...")
            df = pd.read_csv(csv_file_path, encoding='utf-8')

        df = df.where(pd.notnull(df), None)

        print(f"DB('exercise_tier' 테이블)에 {len(df)}개 데이터 삽입 시작...")
        df.to_sql(
            name='exercise_tier',
            con=engine,
            if_exists='replace',
            index=False
        )
        print("✅ 성공적으로 저장되었습니다!")

    except Exception as e:
        print(f"에러 발생: {e}")
        print("팁: pip install openpyxl 명령어를 실행해 보세요.")

if __name__ == "__main__":
    import_data()
