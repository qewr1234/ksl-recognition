"""
MySQL 데이터베이스 (수어/음성 인식 시스템)
- 6개 테이블, 외래키, 3정규형
- 테이블당 100개 이상 Tuple
"""

import mysql.connector
from mysql.connector import Error
from datetime import datetime
import hashlib

# ============================================================
# MySQL 연결 설정 (환경변수 또는 기본값 사용)
# ============================================================
import os

DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'sign_admin'),
    'password': os.environ.get('DB_PASSWORD', 'your_password'),
    'database': os.environ.get('DB_NAME', 'sign_speech_db')
}


# ============================================================
# 연결 함수
# ============================================================
def get_connection():
    """MySQL 연결"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"❌ MySQL 연결 실패: {e}")
        return None


# ============================================================
# 초기화 함수
# ============================================================
def init_db():
    """DB + 테이블 + 샘플 데이터 초기화"""
    create_database()
    create_tables()
    insert_sample_data()
    print("\n✅ DB 초기화 완료!")


def create_database():
    """데이터베이스 생성"""
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        conn.commit()
        cursor.close()
        conn.close()
        print(f"📦 데이터베이스 '{DB_CONFIG['database']}' 준비 완료")
    except Error as e:
        print(f"❌ 데이터베이스 생성 실패: {e}")


def create_tables():
    """테이블 생성 (6개)"""
    conn = get_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # 1. 사용자 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            email VARCHAR(100) NOT NULL UNIQUE,
            password_hash VARCHAR(256) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_username (username)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    
    # 2. 사용자 설정 테이블 (1:1 관계)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            setting_id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL UNIQUE,
            font_size INT DEFAULT 26,
            theme VARCHAR(20) DEFAULT 'dark',
            language VARCHAR(10) DEFAULT 'ko',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
                ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    
    # 3. 세션 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            session_uuid VARCHAR(36) NOT NULL UNIQUE,
            session_name VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
                ON DELETE CASCADE ON UPDATE CASCADE,
            INDEX idx_uuid (session_uuid)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    
    # 4. 카테고리 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INT AUTO_INCREMENT PRIMARY KEY,
            category_name VARCHAR(50) NOT NULL UNIQUE,
            description VARCHAR(200),
            icon VARCHAR(50),
            display_order INT DEFAULT 0
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    
    # 5. 문장 테이블 (수어 사전)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS phrases (
            phrase_id INT AUTO_INCREMENT PRIMARY KEY,
            category_id INT NOT NULL,
            phrase_text VARCHAR(500) NOT NULL,
            sign_video_url VARCHAR(500),
            use_count INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
                ON DELETE RESTRICT ON UPDATE CASCADE,
            INDEX idx_category (category_id),
            INDEX idx_use_count (use_count DESC)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    
    # 6. 대화 기록 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INT AUTO_INCREMENT PRIMARY KEY,
            session_id INT NOT NULL,
            user_id INT NOT NULL,
            input_type ENUM('speech', 'sign', 'text', 'phrase') NOT NULL,
            input_text TEXT NOT NULL,
            confidence DECIMAL(5,4),
            duration DECIMAL(6,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                ON DELETE CASCADE ON UPDATE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
                ON DELETE CASCADE ON UPDATE CASCADE,
            INDEX idx_session (session_id),
            INDEX idx_created (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("📋 테이블 6개 생성 완료")


def insert_sample_data():
    """샘플 데이터 삽입 (100개 이상)"""
    conn = get_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # 이미 데이터 있으면 스킵
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] > 0:
        print("📊 샘플 데이터 이미 존재")
        cursor.close()
        conn.close()
        return
    
    # 1. 카테고리 (10개)
    categories = [
        ('인사', '인사말', '👋', 1),
        ('응답', '예/아니오', '✅', 2),
        ('요청', '부탁 표현', '🙏', 3),
        ('감정', '감정 표현', '😊', 4),
        ('질문', '질문 표현', '❓', 5),
        ('시간', '시간 표현', '🕐', 6),
        ('장소', '장소 표현', '📍', 7),
        ('음식', '음식 표현', '🍚', 8),
        ('교통', '교통 표현', '🚌', 9),
        ('일상', '일상 대화', '💬', 10),
    ]
    cursor.executemany(
        "INSERT INTO categories (category_name, description, icon, display_order) VALUES (%s,%s,%s,%s)",
        categories
    )
    
    # 2. 문장 (150개)
# 2. 문장 (250개 이상)
    phrases = [
        # ===== 인사 (30개) =====
        (1, '안녕하세요'), (1, '안녕히 가세요'), (1, '안녕히 계세요'),
        (1, '만나서 반갑습니다'), (1, '오랜만이에요'), (1, '처음 뵙겠습니다'),
        (1, '좋은 아침이에요'), (1, '좋은 하루 되세요'), (1, '잘 자요'),
        (1, '다음에 또 만나요'), (1, '조심히 가세요'), (1, '반가워요'),
        (1, '어서오세요'), (1, '잘 가요'), (1, '또 봐요'),
        (1, '건강하세요'), (1, '행복하세요'), (1, '좋은 주말 되세요'),
        (1, '새해 복 많이 받으세요'), (1, '생일 축하합니다'),
        (1, '환영합니다'), (1, '처음 뵙겠습니다'), (1, '잘 지내셨어요?'),
        (1, '오래간만이네요'), (1, '보고 싶었어요'), (1, '좋은 저녁이에요'),
        (1, '편안한 밤 되세요'), (1, '내일 봐요'), (1, '좋은 꿈 꾸세요'),
        (1, '수고하셨습니다'),
        
        # ===== 응답 (25개) =====
        (2, '네'), (2, '아니요'), (2, '알겠습니다'), (2, '모르겠습니다'),
        (2, '좋아요'), (2, '싫어요'), (2, '괜찮아요'), (2, '그래요'),
        (2, '맞아요'), (2, '아니에요'), (2, '동의합니다'), (2, '반대합니다'),
        (2, '확실해요'), (2, '아마도요'), (2, '글쎄요'), (2, '당연하죠'),
        (2, '물론이죠'), (2, '정말요?'), (2, '그럴 수도 있어요'), (2, '생각해 볼게요'),
        (2, '잘 모르겠어요'), (2, '그렇군요'), (2, '이해했어요'), (2, '기억할게요'),
        (2, '노력할게요'),
        
        # ===== 요청 (30개) =====
        (3, '잠시만요'), (3, '다시 말해주세요'), (3, '천천히 말해주세요'),
        (3, '도와주세요'), (3, '부탁드립니다'), (3, '죄송합니다'),
        (3, '실례합니다'), (3, '저기요'), (3, '잠깐만 기다려주세요'),
        (3, '크게 말해주세요'), (3, '적어주세요'), (3, '보여주세요'),
        (3, '가르쳐주세요'), (3, '설명해주세요'), (3, '확인해주세요'),
        (3, '연락주세요'), (3, '전화해주세요'), (3, '메시지 보내주세요'),
        (3, '사진 찍어주세요'), (3, '추천해주세요'), (3, '한 번 더요'),
        (3, '수정해주세요'), (3, '취소해주세요'), (3, '예약해주세요'),
        (3, '포장해주세요'), (3, '배달해주세요'), (3, '계산해주세요'),
        (3, '영수증 주세요'), (3, '봉투 주세요'), (3, '물 주세요'),
        
        # ===== 감정 (25개) =====
        (4, '기뻐요'), (4, '슬퍼요'), (4, '화나요'), (4, '무서워요'),
        (4, '놀랐어요'), (4, '걱정돼요'), (4, '피곤해요'), (4, '배고파요'),
        (4, '목말라요'), (4, '졸려요'), (4, '아파요'), (4, '행복해요'),
        (4, '신나요'), (4, '편해요'), (4, '불편해요'), (4, '속상해요'),
        (4, '답답해요'), (4, '외로워요'), (4, '부끄러워요'), (4, '창피해요'),
        (4, '긴장돼요'), (4, '설레요'), (4, '지루해요'), (4, '심심해요'),
        (4, '후회해요'),
        
        # ===== 질문 (30개) =====
        (5, '이게 뭐예요?'), (5, '어디예요?'), (5, '언제예요?'), (5, '왜요?'),
        (5, '어떻게 해요?'), (5, '누구예요?'), (5, '얼마예요?'), (5, '몇 시예요?'),
        (5, '괜찮으세요?'), (5, '뭐 드실래요?'), (5, '어디 가세요?'),
        (5, '도움이 필요하세요?'), (5, '이해하셨어요?'), (5, '질문 있으세요?'),
        (5, '시간 있으세요?'), (5, '뭐 하세요?'), (5, '어디서 왔어요?'),
        (5, '이름이 뭐예요?'), (5, '몇 살이에요?'), (5, '직업이 뭐예요?'),
        (5, '취미가 뭐예요?'), (5, '결혼하셨어요?'), (5, '자녀가 있으세요?'),
        (5, '어디 사세요?'), (5, '연락처가 어떻게 되세요?'), (5, '무슨 일이에요?'),
        (5, '왜 그래요?'), (5, '뭐가 문제예요?'), (5, '도와드릴까요?'),
        (5, '같이 할래요?'),
        
        # ===== 시간 (20개) =====
        (6, '지금'), (6, '오늘'), (6, '내일'), (6, '어제'), (6, '모레'),
        (6, '이번 주'), (6, '다음 주'), (6, '지난 주'), (6, '이번 달'),
        (6, '아침'), (6, '점심'), (6, '저녁'), (6, '밤'), (6, '새벽'),
        (6, '잠깐만'), (6, '나중에'), (6, '곧'), (6, '항상'), (6, '가끔'),
        (6, '매일'),
        
        # ===== 장소 (25개) =====
        (7, '여기'), (7, '저기'), (7, '거기'), (7, '집'), (7, '학교'),
        (7, '회사'), (7, '병원'), (7, '은행'), (7, '마트'), (7, '편의점'),
        (7, '식당'), (7, '카페'), (7, '역'), (7, '공항'), (7, '화장실'),
        (7, '주차장'), (7, '공원'), (7, '도서관'), (7, '영화관'), (7, '백화점'),
        (7, '약국'), (7, '우체국'), (7, '경찰서'), (7, '소방서'), (7, '시청'),
        
        # ===== 음식 (30개) =====
        (8, '물 주세요'), (8, '밥 주세요'), (8, '커피 주세요'),
        (8, '맛있어요'), (8, '맵지 않게 해주세요'), (8, '계산해주세요'),
        (8, '메뉴판 주세요'), (8, '포장해주세요'), (8, '여기서 먹을게요'),
        (8, '배불러요'), (8, '더 주세요'), (8, '덜 주세요'),
        (8, '따뜻하게 해주세요'), (8, '차갑게 해주세요'), (8, '추천해주세요'),
        (8, '소금 빼주세요'), (8, '설탕 빼주세요'), (8, '얼음 빼주세요'),
        (8, '많이 주세요'), (8, '조금만 주세요'), (8, '리필해주세요'),
        (8, '테이크아웃이요'), (8, '배달되나요?'), (8, '예약했어요'),
        (8, '2인분이요'), (8, '반반으로 해주세요'), (8, '곱빼기요'),
        (8, '국물 많이요'), (8, '면 추가요'), (8, '밥 추가요'),
        
        # ===== 교통 (20개) =====
        (9, '버스 정류장 어디예요?'), (9, '지하철역 어디예요?'),
        (9, '택시 불러주세요'), (9, '여기서 내릴게요'), (9, '얼마나 걸려요?'),
        (9, '길을 잃었어요'), (9, '이 버스 어디 가요?'),
        (9, '막차 몇 시예요?'), (9, '첫차 몇 시예요?'), (9, '환승이에요'),
        (9, '요금이 얼마예요?'), (9, '카드 되나요?'), (9, '현금만 되나요?'),
        (9, '자리 있어요?'), (9, '다음 정류장이 어디예요?'),
        (9, '어디서 갈아타요?'), (9, '몇 번 버스예요?'), (9, '몇 호선이에요?'),
        (9, '출구가 어디예요?'), (9, '엘리베이터 어디예요?'),
        
        # ===== 일상 (25개) =====
        (10, '오늘 날씨 좋네요'), (10, '비가 와요'), (10, '덥네요'),
        (10, '춥네요'), (10, '저는 학생이에요'), (10, '저는 직장인이에요'),
        (10, '취미가 뭐예요?'), (10, '주말에 뭐 해요?'), (10, '영화 보러 갈래요?'),
        (10, '산책하러 가요'), (10, '운동하러 가요'), (10, '쇼핑하러 가요'),
        (10, '여행 가고 싶어요'), (10, '공부 중이에요'), (10, '일하는 중이에요'),
        (10, '바빠요'), (10, '한가해요'), (10, '쉬고 싶어요'),
        (10, '놀러 가요'), (10, '집에 가요'), (10, '밥 먹으러 가요'),
        (10, '커피 마실래요?'), (10, '술 마실래요?'), (10, '노래방 갈래요?'),
        (10, '게임할래요?'),
    ]
    cursor.executemany(
        "INSERT INTO phrases (category_id, phrase_text) VALUES (%s, %s)",
        phrases
    )
    
    # 3. 사용자 (110명)
    users = []
    for i in range(1, 111):
        users.append((
            f'user{i:03d}',
            f'user{i:03d}@test.com',
            hashlib.sha256(f'pass{i}'.encode()).hexdigest()
        ))
    cursor.executemany(
        "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
        users
    )
    
    # 게스트 사용자 추가
    cursor.execute(
        "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
        ('guest', 'guest@local', hashlib.sha256('guest'.encode()).hexdigest())
    )
    
    # 4. 사용자 설정 (111개)
    cursor.execute("SELECT user_id FROM users")
    user_ids = [row[0] for row in cursor.fetchall()]
    
    settings = [(uid, 24 + (uid % 10), 'dark' if uid % 2 == 0 else 'light') 
                for uid in user_ids]
    cursor.executemany(
        "INSERT INTO user_settings (user_id, font_size, theme) VALUES (%s, %s, %s)",
        settings
    )
    
    # 5. 세션 (150개)
    import uuid
    sessions = []
    for i in range(150):
        uid = user_ids[i % len(user_ids)]
        sessions.append((uid, str(uuid.uuid4()), f'세션{i+1}'))
    cursor.executemany(
        "INSERT INTO sessions (user_id, session_uuid, session_name) VALUES (%s, %s, %s)",
        sessions
    )
    
    # 6. 대화 기록 (300개)
    cursor.execute("SELECT session_id, user_id FROM sessions")
    sess_list = cursor.fetchall()
    
    cursor.execute("SELECT phrase_text FROM phrases")
    phrase_list = [row[0] for row in cursor.fetchall()]
    
    conversations = []
    types = ['speech', 'sign', 'text', 'phrase']
    for i in range(300):
        sess = sess_list[i % len(sess_list)]
        text = phrase_list[i % len(phrase_list)]
        input_type = types[i % 4]
        conf = 0.85 + (i % 15) * 0.01
        dur = 1.0 + (i % 20) * 0.1
        conversations.append((sess[0], sess[1], input_type, text, conf, dur))
    
    cursor.executemany(
        "INSERT INTO conversations (session_id, user_id, input_type, input_text, confidence, duration) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        conversations
    )
    
    conn.commit()
    cursor.close()
    conn.close()
    print("📊 샘플 데이터 삽입 완료")


# ============================================================
# API용 함수들
# ============================================================

def save_conversation(session_id: str, text: str, input_type: str = "speech",
                     confidence: float = None, duration: float = None):
    """대화 저장"""
    conn = get_connection()
    if not conn:
        return None
    
    cursor = conn.cursor()
    
    # session_id가 uuid면 조회, 아니면 게스트 세션 생성
    cursor.execute(
        "SELECT session_id, user_id FROM sessions WHERE session_uuid = %s",
        (session_id,)
    )
    result = cursor.fetchone()
    
    if not result:
        # 게스트 세션 생성
        cursor.execute("SELECT user_id FROM users WHERE username = 'guest'")
        guest = cursor.fetchone()
        if guest:
            import uuid
            new_uuid = session_id if len(session_id) > 8 else str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO sessions (user_id, session_uuid, session_name) VALUES (%s, %s, %s)",
                (guest[0], new_uuid, '웹 세션')
            )
            conn.commit()
            session_db_id = cursor.lastrowid
            user_id = guest[0]
        else:
            cursor.close()
            conn.close()
            return None
    else:
        session_db_id, user_id = result
    
    # 대화 저장
    cursor.execute(
        "INSERT INTO conversations (session_id, user_id, input_type, input_text, confidence, duration) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (session_db_id, user_id, input_type, text, confidence, duration)
    )
    
    conversation_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    conn.close()
    
    return conversation_id


def get_conversations(session_id: str, limit: int = 50):
    """대화 기록 조회"""
    conn = get_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT c.input_text, c.input_type, c.confidence, c.created_at
        FROM conversations c
        JOIN sessions s ON c.session_id = s.session_id
        WHERE s.session_uuid = %s
        ORDER BY c.created_at DESC
        LIMIT %s
    """, (session_id, limit))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return results


def get_phrases(category_name: str = None):
    """문장 목록 조회"""
    conn = get_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    
    if category_name:
        cursor.execute("""
            SELECT p.phrase_text, c.category_name, p.use_count
            FROM phrases p
            JOIN categories c ON p.category_id = c.category_id
            WHERE c.category_name = %s
            ORDER BY p.use_count DESC
        """, (category_name,))
    else:
        cursor.execute("""
            SELECT p.phrase_text, c.category_name, p.use_count
            FROM phrases p
            JOIN categories c ON p.category_id = c.category_id
            ORDER BY c.display_order, p.use_count DESC
        """)
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return results


def get_categories():
    """카테고리 목록"""
    conn = get_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT category_id, category_name, description, icon, display_order
        FROM categories ORDER BY display_order
    """)
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return results


def get_stats():
    """DB 통계"""
    conn = get_connection()
    if not conn:
        return {"error": "DB 연결 실패"}
    
    cursor = conn.cursor()
    
    stats = {}
    tables = ['users', 'user_settings', 'sessions', 'categories', 'phrases', 'conversations']
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    return {
        "stats": stats,
        "total_tuples": sum(stats.values())
    }


def increment_phrase_count(phrase_text: str):
    """문장 사용 횟수 증가"""
    conn = get_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE phrases SET use_count = use_count + 1 WHERE phrase_text = %s",
        (phrase_text,)
    )
    conn.commit()
    cursor.close()
    conn.close()


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("🗄️ MySQL 데이터베이스 초기화")
    print("=" * 50)
    
    init_db()
    
    # 통계 출력
    stats = get_stats()
    print("\n📊 테이블별 Tuple 수:")
    print("-" * 30)
    for table, count in stats["stats"].items():
        status = "✅" if count >= 100 else "📌"
        print(f"  {status} {table}: {count}개")
    print("-" * 30)
    print(f"  📈 총: {stats['total_tuples']}개")
    print("=" * 50)