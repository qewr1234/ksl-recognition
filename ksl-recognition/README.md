# 🤟 Korean Sign Language & Speech Recognition System

한국어 지문자(수어 자모) 인식 및 음성 인식 통합 시스템입니다.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ 주요 기능

### 🖐️ 지문자 인식 (Fingerspelling Recognition)
- **ST-GCN** (Spatial Temporal Graph Convolutional Network) 기반 동적 제스처 인식
- **RandomForest / KNN** 폴백 모델 지원
- MediaPipe 기반 실시간 손 랜드마크 추출
- **한글 자동 조합** (초성/중성/종성 → 완성형 한글)
- 31개 지문자 클래스 지원 (자음 14개 + 모음 17개)

### 🎤 음성 인식 (Speech Recognition)
- OpenAI **Whisper** 모델 기반
- 직접 구현한 **디지털 신호처리(DSP)** 모듈:
  - Pre-emphasis Filter (고역통과 필터)
  - Energy-based VAD (Voice Activity Detection)
  - Zero Crossing Rate 분석
  - Spectral Subtraction 노이즈 제거
  - Wiener Filter
  - Bandpass Filter (300Hz~3400Hz)
- **실시간 WebSocket** 음성 인식 (2초 침묵 시 자동 인식)

### 🗄️ 데이터베이스
- MySQL 기반 6개 테이블 (3정규형 준수)
- 사용자, 세션, 대화 기록, 문장 사전 관리
- 250+ 한국어 문장 사전 내장

## 📁 프로젝트 구조

```
ksl-recognition/
├── server.py              # FastAPI 메인 서버
├── database.py            # MySQL 데이터베이스
├── requirements.txt       # 의존성
├── models/
│   ├── __init__.py
│   ├── sign_model.py      # 지문자 인식 (ST-GCN/RF/KNN)
│   ├── speech_model.py    # 음성 인식 (Whisper + DSP)
│   └── realtime_speech.py # 실시간 음성 처리
├── scripts/
│   ├── collect_data.py    # 데이터 수집 도구
│   └── download_whisper.py # Whisper 모델 다운로드
└── data/
    └── fingerspell_data.csv # 지문자 학습 데이터
```

## 🚀 설치 및 실행

### 1. 요구사항

- Python 3.9+
- MySQL 8.0+
- 웹캠 (지문자 인식용)
- 마이크 (음성 인식용)

### 2. 설치

```bash
# 레포지토리 클론
git clone https://github.com/username/ksl-recognition.git
cd ksl-recognition

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. Whisper 모델 다운로드

```bash
python scripts/download_whisper.py
```

사용 가능한 모델 크기:
- `tiny`: 가장 빠름 (~39M)
- `base`: 빠름 (~74M)
- `small`: 균형 (~244M)
- `medium`: 정확 (~769M) ✅ 권장
- `large-v3`: 가장 정확 (~1.5G)

### 4. MySQL 설정

```sql
-- MySQL에서 사용자 생성
CREATE USER 'sign_admin'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON sign_speech_db.* TO 'sign_admin'@'localhost';
FLUSH PRIVILEGES;
```

`database.py`의 `DB_CONFIG`를 환경에 맞게 수정:

```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'sign_admin',
    'password': 'your_password',
    'database': 'sign_speech_db'
}
```

### 5. 서버 실행

```bash
python server.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

## 📖 API 엔드포인트

### 지문자 인식

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/sign/predict` | 이미지에서 지문자 인식 |
| POST | `/sign/reset` | 조합 상태 초기화 |
| POST | `/sign/backspace` | 마지막 입력 삭제 |
| POST | `/sign/space` | 공백 추가 |
| POST | `/sign/submit` | 완성된 텍스트 DB 저장 |

### 음성 인식

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/speech/predict` | 오디오에서 음성 인식 |
| WS | `/ws/speech` | 실시간 음성 인식 WebSocket |

### 데이터베이스

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/categories` | 문장 카테고리 목록 |
| GET | `/phrases` | 문장 사전 조회 |
| GET | `/conversations` | 대화 기록 조회 |
| GET | `/db/stats` | DB 통계 |
| GET | `/health` | 서버 상태 확인 |

## 🖐️ 지문자 매핑

### 자음 (14개)
| 라벨 | 지문자 | 라벨 | 지문자 |
|------|--------|------|--------|
| 0 | ㄱ | 7 | ㅇ |
| 1 | ㄴ | 8 | ㅈ |
| 2 | ㄷ | 9 | ㅊ |
| 3 | ㄹ | 10 | ㅋ |
| 4 | ㅁ | 11 | ㅌ |
| 5 | ㅂ | 12 | ㅍ |
| 6 | ㅅ | 13 | ㅎ |

### 모음 (17개)
| 라벨 | 지문자 | 라벨 | 지문자 |
|------|--------|------|--------|
| 14 | ㅏ | 22 | ㅡ |
| 15 | ㅑ | 23 | ㅣ |
| 16 | ㅓ | 24 | ㅐ |
| 17 | ㅕ | 25 | ㅔ |
| 18 | ㅗ | 26 | ㅚ |
| 19 | ㅛ | 27 | ㅟ |
| 20 | ㅜ | 28 | ㅢ |
| 21 | ㅠ | 29 | ㅒ |
|    |    | 30 | ㅖ |

## 🔧 데이터 수집

지문자 데이터 수집 도구 사용법:

```bash
python scripts/collect_data.py
```

**조작법:**
- `0~9`, `a~v`: 라벨 선택
- `SPACE`: 녹화 시작/중지
- `s`: 데이터 저장
- `q`: 종료

## 🏗️ 아키텍처

### 지문자 인식 파이프라인

```
웹캠 → MediaPipe (손 랜드마크) → 특징 추출 → 모델 추론 → 다수결 → 한글 조합
                                    ↓
                            ST-GCN / RF / KNN
```

### 음성 인식 파이프라인

```
마이크 → DSP 전처리 → VAD → Whisper → 텍스트 후처리 → 결과
           ↓
    Pre-emphasis
    Bandpass Filter
    Noise Reduction
```

## 📊 성능

| 모델 | 정확도 | 속도 |
|------|--------|------|
| ST-GCN | ~95% | 실시간 |
| RandomForest | ~92% | 실시간 |
| KNN (폴백) | ~85% | 실시간 |
| Whisper (medium) | ~95% | 준실시간 |

## 🛠️ 기술 스택

- **Backend**: FastAPI, WebSocket
- **ML/DL**: PyTorch, scikit-learn
- **Computer Vision**: OpenCV, MediaPipe
- **Speech**: Transformers (Whisper)
- **Database**: MySQL
- **Signal Processing**: NumPy (직접 구현)

## 📝 라이선스

MIT License

## 🙏 감사의 글

- [MediaPipe](https://google.github.io/mediapipe/) - 손 랜드마크 추출
- [OpenAI Whisper](https://github.com/openai/whisper) - 음성 인식
- [ST-GCN](https://github.com/yysijie/st-gcn) - 그래프 기반 행동 인식

---

**Made with ❤️ for Korean Sign Language Community**
