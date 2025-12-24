"""
지문자 + 음성 인식 API 서버 (실시간 음성 인식 포함)
- 지문자: 엔터 시 DB 저장
- 음성: 실시간 WebSocket + 2초 침묵 시 자동 저장

실행: python server.py
접속: http://localhost:8000
"""

from fastapi import FastAPI, UploadFile, File, Form, Query, Request
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
import uuid
import asyncio
import base64
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

app = FastAPI(title="지문자/음성 인식 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

speech_model = None
sign_model = None


# ================================================================
# 실시간 음성 인식 클래스들 (내장)
# ================================================================

@dataclass
class RealtimeConfig:
    sample_rate: int = 16000
    chunk_duration: float = 0.1
    silence_threshold: float = 2.0
    min_speech_duration: float = 0.5
    energy_threshold: float = 0.01
    max_buffer_duration: float = 30.0


class EnergyVAD:
    """에너지 기반 VAD"""
    def __init__(self, energy_threshold=0.01, smoothing_window=5):
        self.energy_threshold = energy_threshold
        self.energy_history = deque(maxlen=smoothing_window)
    
    def is_speech(self, audio_chunk):
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        self.energy_history.append(energy)
        return np.mean(self.energy_history) > self.energy_threshold
    
    def reset(self):
        self.energy_history.clear()


class RealtimeSpeechRecognizer:
    """실시간 음성 인식기"""
    def __init__(self, config=None, dsp=None, recognizer=None):
        self.config = config or RealtimeConfig()
        self.audio_buffer = []
        self.vad = EnergyVAD(energy_threshold=self.config.energy_threshold)
        self.last_speech_time = None
        self.speech_started = False
        self.is_speaking = False
        self.dsp = dsp
        self.recognizer = recognizer
        self.on_final_result = None
        self.on_status_change = None
        self.is_running = False
    
    def start(self):
        self.is_running = True
        self.reset()
        if self.on_status_change:
            self.on_status_change("listening")
    
    def stop(self):
        self.is_running = False
        result = None
        if self.audio_buffer and self.speech_started:
            result = self._process_buffer()
        self.reset()
        if self.on_status_change:
            self.on_status_change("stopped")
        return result
    
    def reset(self):
        self.audio_buffer = []
        self.last_speech_time = None
        self.speech_started = False
        self.is_speaking = False
        self.vad.reset()
    
    def process_chunk(self, audio_chunk):
        if not self.is_running:
            return None
        
        current_time = time.time()
        self.audio_buffer.append(audio_chunk)
        
        # 버퍼 최대 길이 체크
        buffer_duration = len(self.audio_buffer) * self.config.chunk_duration
        if buffer_duration > self.config.max_buffer_duration:
            return self._process_and_reset()
        
        # VAD
        is_speech = self.vad.is_speech(audio_chunk)
        
        if is_speech:
            self.last_speech_time = current_time
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_started = True
                if self.on_status_change:
                    self.on_status_change("speaking")
        else:
            if self.is_speaking:
                self.is_speaking = False
                if self.on_status_change:
                    self.on_status_change("silence")
            
            # 2초 침묵 감지
            if (self.speech_started and 
                self.last_speech_time and 
                current_time - self.last_speech_time >= self.config.silence_threshold):
                return self._process_and_reset()
        
        return None
    
    def _process_and_reset(self):
        result = self._process_buffer()
        self.audio_buffer = []
        self.last_speech_time = None
        self.speech_started = False
        self.is_speaking = False
        self.vad.reset()
        if self.on_status_change:
            self.on_status_change("listening")
        return result
    
    def _process_buffer(self):
        if not self.audio_buffer or not self.recognizer:
            return None
        
        audio = np.concatenate(self.audio_buffer)
        duration = len(audio) / self.config.sample_rate
        
        if duration < self.config.min_speech_duration:
            return None
        
        # Pre-emphasis
        if self.dsp:
            processed_audio = self.dsp.pre_emphasis(audio, alpha=0.97)
        else:
            processed_audio = audio
        
        # Whisper 인식
        result = self.recognizer.predict(processed_audio, use_dsp=False)
        
        if result.get("text"):
            if self.on_final_result:
                self.on_final_result(result["text"])
            return {"text": result["text"], "duration": duration}
        return None


# ================================================================
# 서버 시작
# ================================================================

@app.on_event("startup")
async def startup():
    global speech_model, sign_model
    
    print("=" * 50)
    print("Server Starting")
    print("=" * 50)
    
    # DB 초기화
    try:
        from database import init_db
        init_db()
        print("[OK] Database initialized")
    except Exception as e:
        print(f"[WARN] DB: {e}")
    
    # 음성 모델
    try:
        from models.speech_model import SpeechRecognizer
        speech_model = SpeechRecognizer()
        print("[OK] Speech model loaded")
    except Exception as e:
        print(f"[WARN] Speech: {e}")
    
    # 지문자 모델
    try:
        from models.sign_model import SignRecognizer
        sign_model = SignRecognizer()
        print(f"[OK] Sign model loaded ({sign_model.model_type})")
    except Exception as e:
        print(f"[ERROR] Sign: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 50)
    model_type = sign_model.model_type if sign_model else "none"
    print(f"Sign model: {sign_model is not None} ({model_type})")
    print(f"Speech model: {speech_model is not None}")
    print("Server Ready! → http://localhost:8000")
    print("=" * 50)


# ================================================================
# 실시간 음성 인식 WebSocket
# ================================================================

active_speech_connections: dict = {}

@app.websocket("/ws/speech/{session_id}")
async def websocket_realtime_speech(websocket: WebSocket, session_id: str):
    """
    실시간 음성 인식 WebSocket
    
    클라이언트 → 서버:
        - {"type": "start", "user_id": 1}
        - {"type": "audio", "data": "<base64 PCM 16-bit>"}
        - {"type": "stop"}
    
    서버 → 클라이언트:
        - {"type": "status", "status": "listening|speaking|silence|stopped"}
        - {"type": "result", "text": "인식된 텍스트", "saved": true}
    """
    await websocket.accept()
    print(f"\n🔌 WebSocket 연결: session_id={session_id}")
    
    if speech_model is None:
        await websocket.send_json({"type": "error", "message": "음성 모델 없음"})
        await websocket.close()
        return
    
    # 실시간 인식기 초기화
    config = RealtimeConfig(
        sample_rate=16000,
        silence_threshold=2.0,
        min_speech_duration=0.5,
        energy_threshold=0.01
    )
    recognizer = RealtimeSpeechRecognizer(
        config=config,
        dsp=speech_model.dsp,
        recognizer=speech_model
    )
    
    user_id = 1
    active_speech_connections[session_id] = websocket
    
    # 콜백 함수들
    async def send_json_safe(data: dict):
        try:
            await websocket.send_json(data)
        except:
            pass
    
    async def on_result(text: str):
        saved = False
        # DB 저장
        try:
            from database import save_conversation
            save_conversation(session_id, text, "speech")
            saved = True
            print(f"  💾 DB 저장: {text}")
        except Exception as e:
            print(f"  ⚠️ DB 저장 실패: {e}")
        
        await send_json_safe({
            "type": "result",
            "text": text,
            "saved": saved
        })
    
    def sync_status(status: str):
        asyncio.create_task(send_json_safe({"type": "status", "status": status}))
    
    def sync_result(text: str):
        asyncio.create_task(on_result(text))
    
    recognizer.on_status_change = sync_status
    recognizer.on_final_result = sync_result
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "start":
                user_id = data.get("user_id", 1)
                recognizer.start()
                print(f"  🎤 실시간 인식 시작")
                
            elif msg_type == "audio":
                if recognizer.is_running:
                    audio_base64 = data.get("data", "")
                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_int16.astype(np.float32) / 32768.0
                        recognizer.process_chunk(audio_float)
                
            elif msg_type == "stop":
                final = recognizer.stop()
                if final and final.get("text"):
                    await on_result(final["text"])
                await send_json_safe({"type": "status", "status": "stopped"})
                print("  🛑 실시간 인식 종료")
                break
                
    except WebSocketDisconnect:
        print(f"  🔌 WebSocket 연결 종료: {session_id}")
    except Exception as e:
        print(f"  ⚠️ WebSocket 에러: {e}")
        await send_json_safe({"type": "error", "message": str(e)})
    finally:
        if recognizer.is_running:
            recognizer.stop()
        if session_id in active_speech_connections:
            del active_speech_connections[session_id]


# ================================================================
# 음성 API (기존 배치 방식)
# ================================================================

@app.post("/speech/predict")
async def predict_speech(request: Request, session_id: str = Form(None)):
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        
        if speech_model is None:
            return {"success": False, "text": "", "error": "no model"}
        
        body = await request.body()
        audio_array = np.frombuffer(body, dtype=np.float32)
        
        if len(audio_array) < 8000:
            return {"success": False, "text": "", "error": "too short"}
        
        result = speech_model.predict(audio_array)
        
        if result.get("text"):
            try:
                from database import save_conversation
                save_conversation(session_id, result["text"], "speech")
            except:
                pass
        
        return {"success": True, "text": result.get("text", ""), "session_id": session_id}
    
    except Exception as e:
        return {"success": False, "text": "", "error": str(e)}


# ================================================================
# 지문자 API
# ================================================================

@app.post("/sign/predict")
async def predict_sign(video: UploadFile = File(...), session_id: str = Form(None)):
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        
        if sign_model is None:
            return {
                "success": False, "text": "", "error": "no model",
                "hands": "none", "buffer": 0, "buffer_max": 15,
                "status": "no model", "landmarks": [],
                "current_char": "", "composed_text": "", "model_type": "none"
            }
        
        video_bytes = await video.read()
        result = sign_model.predict(video_bytes)
        
        return {
            "success": True,
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0),
            "status": result.get("status", ""),
            "hands": result.get("hands", "none"),
            "buffer": result.get("buffer", 0),
            "buffer_max": result.get("buffer_max", 15),
            "landmarks": result.get("landmarks", []),
            "current_char": result.get("current_char", ""),
            "composed_text": result.get("composed_text", ""),
            "model_type": result.get("model_type", "unknown"),
            "session_id": session_id
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False, "text": "", "error": str(e),
            "hands": "none", "status": "error", "landmarks": [],
            "current_char": "", "composed_text": "", "model_type": "error"
        }


@app.post("/sign/submit")
async def submit_sign(session_id: str = Form(None)):
    if sign_model is None:
        return {"success": False, "error": "no model"}
    
    composed_text = sign_model.get_composed_text()
    
    if composed_text and composed_text.strip():
        try:
            from database import save_conversation
            save_conversation(session_id or "guest", composed_text.strip(), "sign")
            print(f"  [DB] Saved: {composed_text.strip()}")
        except Exception as e:
            print(f"  [WARN] DB save failed: {e}")
    
    sign_model.reset()
    
    return {
        "success": True,
        "saved_text": composed_text.strip() if composed_text else "",
        "composed_text": ""
    }


@app.post("/sign/reset")
async def reset_sign():
    if sign_model:
        sign_model.reset()
    return {"success": True}


@app.post("/sign/backspace")
async def backspace_sign():
    if sign_model:
        text = sign_model.backspace()
        return {"success": True, "composed_text": text}
    return {"success": False}


@app.post("/sign/space")
async def space_sign():
    if sign_model:
        text = sign_model.add_space()
        return {"success": True, "composed_text": text}
    return {"success": False}


# ================================================================
# DB API
# ================================================================

@app.get("/phrases")
async def list_phrases(category: str = Query(None)):
    try:
        from database import get_phrases
        return {"phrases": get_phrases(category)}
    except Exception as e:
        return {"phrases": [], "error": str(e)}


@app.get("/categories")
async def list_categories():
    try:
        from database import get_categories
        return {"categories": get_categories()}
    except Exception as e:
        return {"categories": [], "error": str(e)}


@app.get("/conversations")
async def list_conversations(session_id: str = Query(...)):
    try:
        from database import get_conversations
        return {"conversations": get_conversations(session_id)}
    except Exception as e:
        return {"conversations": [], "error": str(e)}


@app.post("/phrase/use")
async def use_phrase(phrase_text: str = Form(...), session_id: str = Form(None)):
    try:
        from database import increment_phrase_count, save_conversation
        increment_phrase_count(phrase_text)
        if session_id:
            save_conversation(session_id, phrase_text, "phrase")
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/health")
async def health():
    model_type = sign_model.model_type if sign_model else "none"
    return {
        "status": "ok",
        "speech": speech_model is not None,
        "sign": sign_model is not None,
        "sign_model_type": model_type,
        "realtime_connections": len(active_speech_connections)
    }


# ================================================================
# 웹 UI
# ================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>지문자/음성 인식</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            min-height: 100vh;
            padding: 15px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        
        h1 { text-align: center; margin-bottom: 20px; color: #00ff88; font-size: 24px; }
        
        .tabs { display: flex; justify-content: center; gap: 10px; margin-bottom: 20px; }
        .tab {
            padding: 12px 30px; border: none; border-radius: 25px;
            cursor: pointer; font-size: 15px; font-weight: 600;
            background: rgba(22, 33, 62, 0.9); color: #888;
            transition: all 0.3s; border: 1px solid rgba(0,255,136,0.2);
        }
        .tab:hover { color: white; }
        .tab.active { background: linear-gradient(135deg, #00ff88, #00ccff); color: #1a1a2e; }
        
        .panel { display: none; }
        .panel.active { display: block; }
        
        .card {
            background: rgba(22, 33, 62, 0.95); border-radius: 15px;
            padding: 20px; margin-bottom: 15px;
            border: 1px solid rgba(0,255,136,0.2);
        }
        .card h3 { color: #00ff88; margin-bottom: 15px; font-size: 16px; }
        
        #videoContainer {
            position: relative; width: 100%; aspect-ratio: 4/3;
            background: #000; border-radius: 12px; overflow: hidden;
            border: 2px solid rgba(0,255,136,0.3);
        }
        #video { width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); }
        #overlay {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            pointer-events: none; transform: scaleX(-1);
        }
        
        .current-char {
            position: absolute; top: 15px; right: 15px;
            font-size: 70px; font-weight: bold; color: #00ff88;
            text-shadow: 0 0 25px rgba(0,255,136,0.8); z-index: 10;
        }
        
        .model-badge {
            position: absolute; top: 15px; left: 15px;
            padding: 5px 12px; border-radius: 15px;
            font-size: 12px; font-weight: bold;
            background: rgba(0,0,0,0.7); color: #00ff88;
        }
        
        .status-bar {
            display: flex; justify-content: space-between; align-items: center;
            margin: 12px 0; padding: 10px 12px;
            background: rgba(0,0,0,0.3); border-radius: 8px;
        }
        .hand-icon { font-size: 30px; opacity: 0.3; transition: all 0.3s; }
        .hand-icon.active { opacity: 1; }
        
        .progress-bar {
            flex: 1; height: 8px; background: #333;
            border-radius: 4px; margin: 0 12px; overflow: hidden;
        }
        .progress-level {
            height: 100%; background: linear-gradient(90deg, #00ff88, #00ccff);
            width: 0%; transition: width 0.1s;
        }
        
        .btn-group { display: flex; justify-content: center; gap: 15px; margin: 20px 0; }
        
        .record-btn {
            width: 90px; height: 90px; border-radius: 50%;
            background: linear-gradient(135deg, #00ff88, #00ccff);
            border: none; cursor: pointer; font-size: 15px; font-weight: bold;
            color: #1a1a2e; box-shadow: 0 4px 20px rgba(0,255,136,0.4);
            transition: all 0.3s;
        }
        .record-btn:hover { transform: scale(1.05); }
        .record-btn.recording {
            background: linear-gradient(135deg, #ff4444, #ff6b6b);
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(255,68,68,0.7); }
            50% { box-shadow: 0 0 0 20px rgba(255,68,68,0); }
        }
        
        .action-btn {
            width: 55px; height: 55px; border-radius: 50%;
            border: 2px solid #00ff88; background: transparent;
            color: #00ff88; cursor: pointer; font-size: 22px;
            transition: all 0.3s;
        }
        .action-btn:hover { background: rgba(0,255,136,0.2); }
        
        .submit-btn {
            width: 55px; height: 55px; border-radius: 50%;
            border: 2px solid #00ccff; background: transparent;
            color: #00ccff; cursor: pointer; font-size: 18px;
            transition: all 0.3s;
        }
        .submit-btn:hover { background: rgba(0,204,255,0.2); }
        
        .status-text {
            text-align: center; color: #888; font-size: 14px;
            min-height: 20px; margin: 8px 0;
        }
        .status-text.detecting { color: #ffcc00; }
        .status-text.recognized { color: #00ff88; font-weight: bold; }
        .status-text.error { color: #ff6b6b; }
        .status-text.speaking { color: #00ccff; font-weight: bold; }
        .status-text.silence { color: #ffcc00; }
        
        .composed-box {
            background: linear-gradient(135deg, #0f3460, #1a1a4e);
            border-radius: 12px; padding: 20px; min-height: 80px;
            border: 1px solid rgba(0,255,136,0.3);
        }
        .composed-text {
            font-size: 28px; text-align: center;
            word-break: keep-all; line-height: 1.5;
        }
        .composed-text:empty::before {
            content: "인식된 내용이 여기에 표시됩니다";
            color: #555; font-size: 14px;
        }
        
        .categories { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px; }
        .category-btn {
            padding: 8px 16px; border-radius: 20px;
            border: 1px solid rgba(0,255,136,0.3); background: transparent;
            color: #aaa; cursor: pointer; font-size: 13px; transition: all 0.3s;
        }
        .category-btn:hover { color: white; border-color: #00ff88; }
        .category-btn.active {
            background: rgba(0,255,136,0.2); color: #00ff88; border-color: #00ff88;
        }
        
        .phrases-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 10px; max-height: 300px; overflow-y: auto;
        }
        .phrase-btn {
            padding: 12px 10px; border-radius: 10px;
            border: 1px solid rgba(0,255,136,0.2); background: rgba(0,0,0,0.2);
            color: white; cursor: pointer; font-size: 13px;
            text-align: center; transition: all 0.3s;
        }
        .phrase-btn:hover { background: rgba(0,255,136,0.2); border-color: #00ff88; }
        
        .history-list { max-height: 200px; overflow-y: auto; }
        .history-item {
            padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex; justify-content: space-between; align-items: center;
        }
        .history-type {
            font-size: 10px; padding: 3px 8px; border-radius: 10px; background: #333;
        }
        .history-type.speech { background: #00ff88; color: #1a1a2e; }
        .history-type.sign { background: #ff6b6b; }
        .history-type.phrase { background: #00ccff; color: #1a1a2e; }
        
        .volume-bar { height: 8px; background: #333; border-radius: 4px; margin: 12px 0; overflow: hidden; }
        .volume-level { height: 100%; background: #00ff88; width: 0%; transition: width 0.05s; }
        .result-box { background: #0f3460; border-radius: 10px; padding: 15px; min-height: 60px; margin-top: 12px; }
        .result-text { font-size: 22px; text-align: center; }
        
        .realtime-results {
            background: rgba(0,0,0,0.3); border-radius: 10px; padding: 15px;
            min-height: 100px; max-height: 200px; overflow-y: auto; margin-top: 12px;
        }
        .realtime-item {
            padding: 8px 12px; margin: 5px 0; background: rgba(0,255,136,0.1);
            border-radius: 8px; border-left: 3px solid #00ff88;
        }
        .realtime-item.new { animation: fadeIn 0.3s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
        
        .mode-toggle {
            display: flex; justify-content: center; gap: 10px; margin-bottom: 15px;
        }
        .mode-btn {
            padding: 8px 20px; border-radius: 20px; border: 1px solid #00ff88;
            background: transparent; color: #00ff88; cursor: pointer;
            font-size: 13px; transition: all 0.3s;
        }
        .mode-btn.active {
            background: #00ff88; color: #1a1a2e;
        }
        
        .debug-info { font-size: 10px; color: #444; text-align: center; margin-top: 8px; }
        .keyboard-hint { text-align: center; color: #555; font-size: 11px; margin-top: 10px; }
        .keyboard-hint kbd { background: #333; padding: 2px 6px; border-radius: 3px; margin: 0 3px; }
        .keyboard-hint kbd.enter { background: #00ccff; color: #1a1a2e; }
    </style>
</head>
<body>
    <div class="container">
        <h1>✋ 지문자/음성 인식</h1>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('sign')">🤟 지문자</button>
            <button class="tab" onclick="switchTab('speech')">🎤 음성</button>
            <button class="tab" onclick="switchTab('phrases')">📋 문장</button>
        </div>
        
        <!-- 지문자 패널 -->
        <div id="signPanel" class="panel active">
            <div class="card">
                <h3>📹 카메라</h3>
                <div id="videoContainer">
                    <video id="video" autoplay playsinline muted></video>
                    <canvas id="overlay"></canvas>
                    <div class="current-char" id="currentChar"></div>
                    <div class="model-badge" id="modelBadge">모델: -</div>
                </div>
                
                <div class="status-bar">
                    <span class="hand-icon" id="handIcon">✋</span>
                    <div class="progress-bar">
                        <div class="progress-level" id="bufferLevel"></div>
                    </div>
                    <span id="bufferText">0/15</span>
                </div>
                
                <div class="btn-group">
                    <button class="action-btn" onclick="resetSign()" title="초기화 (ESC)">↺</button>
                    <button class="action-btn" onclick="backspaceSign()" title="삭제 (Backspace)">⌫</button>
                    <button class="record-btn" id="signBtn" onclick="toggleSign()">시작</button>
                    <button class="action-btn" onclick="addSpaceSign()" title="공백 (Space)">␣</button>
                    <button class="submit-btn" onclick="submitSign()" title="저장 (Enter)">⏎</button>
                </div>
                
                <p class="status-text" id="signStatus">시작 버튼을 눌러주세요</p>
                <p class="debug-info" id="debugInfo"></p>
            </div>
            
            <div class="card">
                <h3>📝 인식된 문장</h3>
                <div class="composed-box">
                    <p class="composed-text" id="composedText"></p>
                </div>
                <div class="keyboard-hint">
                    <kbd>Space</kbd> 공백
                    <kbd>Backspace</kbd> 삭제
                    <kbd>ESC</kbd> 초기화
                    <kbd class="enter">Enter</kbd> 저장
                </div>
            </div>
        </div>
        
        <!-- 음성 패널 -->
        <div id="speechPanel" class="panel">
            <div class="card">
                <h3>🎤 음성 인식</h3>
                
                <!-- 모드 선택 -->
                <div class="mode-toggle">
                    <button class="mode-btn active" id="realtimeMode" onclick="setSpeechMode('realtime')">🔴 실시간</button>
                    <button class="mode-btn" id="batchMode" onclick="setSpeechMode('batch')">📁 배치</button>
                </div>
                
                <div class="btn-group">
                    <button class="record-btn" id="speechBtn" onclick="toggleSpeech()">녹음</button>
                </div>
                <p class="status-text" id="speechStatus">버튼을 눌러 말하세요</p>
                <div class="volume-bar"><div class="volume-level" id="volumeLevel"></div></div>
                <p class="status-text" id="timer">0:00</p>
                
                <!-- 실시간 결과 -->
                <div id="realtimeResultsContainer">
                    <h4 style="color:#00ff88; margin: 15px 0 10px; font-size: 14px;">📝 실시간 인식 결과 (2초 침묵 시 자동 저장)</h4>
                    <div class="realtime-results" id="realtimeResults">
                        <p style="color:#555; text-align:center;">인식 결과가 여기에 표시됩니다</p>
                    </div>
                </div>
                
                <!-- 배치 결과 -->
                <div id="batchResultContainer" style="display:none;">
                    <div class="result-box">
                        <p class="result-text" id="speechResult">-</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 문장 패널 -->
        <div id="phrasesPanel" class="panel">
            <div class="card">
                <h3>📂 카테고리</h3>
                <div class="categories" id="categoriesContainer">
                    <button class="category-btn active" onclick="loadPhrases(null, this)">전체</button>
                </div>
            </div>
            <div class="card">
                <h3>💬 자주 쓰는 문장</h3>
                <div class="phrases-grid" id="phrasesContainer">로딩 중...</div>
            </div>
        </div>
        
        <!-- 대화 기록 -->
        <div class="card">
            <h3>📜 대화 기록</h3>
            <div class="history-list" id="historyList">
                <p style="color:#555; text-align:center; padding:20px;">아직 대화 기록이 없습니다</p>
            </div>
        </div>
    </div>

    <script>
        const sessionId = Math.random().toString(36).substr(2, 8);
        let isSignRunning = false;
        let isSpeechRecording = false;
        let signInterval;
        let speechMode = 'realtime';  // 'realtime' or 'batch'
        
        // 배치 모드 변수
        let audioContext, mediaStream, scriptProcessor;
        let audioChunks = [];
        let timerInterval, seconds = 0;
        
        // 실시간 모드 변수
        let websocket = null;
        let realtimeAudioContext = null;
        let realtimeStream = null;
        let realtimeProcessor = null;
        
        const video = document.getElementById('video');
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        const captureCanvas = document.createElement('canvas');
        const captureCtx = captureCanvas.getContext('2d');
        
        const HAND_CONNECTIONS = [
            [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
            [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
            [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]
        ];
        
        // 음성 모드 전환
        function setSpeechMode(mode) {
            speechMode = mode;
            document.getElementById('realtimeMode').classList.toggle('active', mode === 'realtime');
            document.getElementById('batchMode').classList.toggle('active', mode === 'batch');
            document.getElementById('realtimeResultsContainer').style.display = mode === 'realtime' ? 'block' : 'none';
            document.getElementById('batchResultContainer').style.display = mode === 'batch' ? 'block' : 'none';
            
            if (isSpeechRecording) {
                toggleSpeech();  // 녹음 중이면 중지
            }
        }
        
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tab + 'Panel').classList.add('active');
            if (tab === 'phrases') { loadCategories(); loadPhrases(null); }
        }
        
        function drawLandmarks(landmarks) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (!landmarks || !Array.isArray(landmarks) || landmarks.length === 0) return;
            
            landmarks.forEach(hand => {
                if (!hand || !hand.points) return;
                const points = hand.points;
                
                ctx.strokeStyle = '#00ff88'; ctx.lineWidth = 3;
                ctx.shadowColor = '#00ff88'; ctx.shadowBlur = 12;
                
                HAND_CONNECTIONS.forEach(([i, j]) => {
                    if (points[i] && points[j]) {
                        ctx.beginPath();
                        ctx.moveTo(points[i].x * canvas.width, points[i].y * canvas.height);
                        ctx.lineTo(points[j].x * canvas.width, points[j].y * canvas.height);
                        ctx.stroke();
                    }
                });
                
                ctx.shadowBlur = 0;
                points.forEach((p, idx) => {
                    if (p) {
                        ctx.fillStyle = idx === 0 ? '#ffcc00' : '#fff';
                        ctx.beginPath();
                        ctx.arc(p.x * canvas.width, p.y * canvas.height, idx === 0 ? 7 : 4, 0, Math.PI * 2);
                        ctx.fill();
                    }
                });
            });
        }
        
        async function toggleSign() {
            const btn = document.getElementById('signBtn');
            const status = document.getElementById('signStatus');
            
            if (!isSignRunning) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: 'user', width: 640, height: 480 }
                    });
                    video.srcObject = stream;
                    canvas.width = 640; canvas.height = 480;
                    captureCanvas.width = 640; captureCanvas.height = 480;
                    
                    isSignRunning = true;
                    btn.textContent = '중지'; btn.classList.add('recording');
                    status.textContent = '손을 보여주세요'; status.className = 'status-text';
                    signInterval = setInterval(captureAndPredict, 100);
                } catch (e) {
                    status.textContent = '카메라 오류: ' + e.message;
                    status.className = 'status-text error';
                }
            } else {
                clearInterval(signInterval);
                if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                isSignRunning = false;
                btn.textContent = '시작'; btn.classList.remove('recording');
                status.textContent = '중지됨';
                document.getElementById('currentChar').textContent = '';
                document.getElementById('handIcon').classList.remove('active');
            }
        }
        
        async function captureAndPredict() {
            if (!video.videoWidth) return;
            captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
            
            captureCanvas.toBlob(async (blob) => {
                if (!blob) return;
                
                const formData = new FormData();
                formData.append('video', blob, 'frame.jpg');
                formData.append('session_id', sessionId);
                
                try {
                    const res = await fetch('/sign/predict', { method: 'POST', body: formData });
                    const data = await res.json();
                    
                    const modelNames = { 'rf': 'RandomForest', 'knn': 'KNN', 'stgcn': 'ST-GCN' };
                    document.getElementById('modelBadge').textContent = 
                        '모델: ' + (modelNames[data.model_type] || data.model_type);
                    
                    document.getElementById('debugInfo').textContent =
                        `hands: ${data.hands || 'none'}, model: ${data.model_type || 'none'}`;
                    
                    if (data.landmarks) drawLandmarks(data.landmarks);
                    
                    const handIcon = document.getElementById('handIcon');
                    handIcon.classList.toggle('active', data.hands && data.hands !== 'none');
                    
                    const buffer = data.buffer || 0;
                    const bufferMax = data.buffer_max || 15;
                    document.getElementById('bufferLevel').style.width = (buffer / bufferMax * 100) + '%';
                    document.getElementById('bufferText').textContent = buffer + '/' + bufferMax;
                    
                    document.getElementById('currentChar').textContent = data.current_char || '';
                    
                    const statusEl = document.getElementById('signStatus');
                    statusEl.textContent = data.status || '';
                    statusEl.className = 'status-text';
                    if (data.status?.includes('인식 중')) statusEl.classList.add('detecting');
                    if (data.status === 'recognized') statusEl.classList.add('recognized');
                    if (data.error) statusEl.classList.add('error');
                    
                    document.getElementById('composedText').textContent = data.composed_text || '';
                    
                } catch (e) {
                    console.error('API Error:', e);
                    document.getElementById('debugInfo').textContent = 'Error: ' + e.message;
                }
            }, 'image/jpeg', 0.85);
        }
        
        async function submitSign() {
            const composedText = document.getElementById('composedText').textContent;
            
            if (!composedText || !composedText.trim()) {
                document.getElementById('signStatus').textContent = '저장할 내용이 없습니다';
                return;
            }
            
            try {
                const res = await fetch('/sign/submit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: 'session_id=' + sessionId
                });
                const data = await res.json();
                
                if (data.success && data.saved_text) {
                    addHistory(data.saved_text, 'sign');
                    document.getElementById('composedText').textContent = '';
                    document.getElementById('currentChar').textContent = '';
                    document.getElementById('signStatus').textContent = '저장 완료!';
                    document.getElementById('signStatus').className = 'status-text recognized';
                    
                    setTimeout(() => {
                        document.getElementById('signStatus').textContent = '손을 보여주세요';
                        document.getElementById('signStatus').className = 'status-text';
                    }, 1500);
                }
            } catch (e) {
                console.error('Submit error:', e);
                document.getElementById('signStatus').textContent = '저장 실패';
                document.getElementById('signStatus').className = 'status-text error';
            }
        }
        
        async function resetSign() {
            await fetch('/sign/reset', { method: 'POST' });
            document.getElementById('composedText').textContent = '';
            document.getElementById('currentChar').textContent = '';
            document.getElementById('signStatus').textContent = '초기화됨';
        }
        
        async function backspaceSign() {
            const res = await fetch('/sign/backspace', { method: 'POST' });
            const data = await res.json();
            if (data.success) document.getElementById('composedText').textContent = data.composed_text || '';
        }
        
        async function addSpaceSign() {
            const res = await fetch('/sign/space', { method: 'POST' });
            const data = await res.json();
            if (data.success) document.getElementById('composedText').textContent = data.composed_text || '';
        }
        
        // ================================================================
        // 음성 녹음 (모드에 따라 분기)
        // ================================================================
        
        async function toggleSpeech() {
            if (speechMode === 'realtime') {
                await toggleRealtimeSpeech();
            } else {
                await toggleBatchSpeech();
            }
        }
        
        // ================================================================
        // 실시간 음성 인식 (WebSocket)
        // ================================================================
        
        async function toggleRealtimeSpeech() {
            const btn = document.getElementById('speechBtn');
            const status = document.getElementById('speechStatus');
            
            if (!isSpeechRecording) {
                try {
                    // WebSocket 연결
                    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                    websocket = new WebSocket(`${wsProtocol}//${location.host}/ws/speech/${sessionId}`);
                    
                    websocket.onopen = () => {
                        console.log('WebSocket 연결됨');
                        websocket.send(JSON.stringify({ type: 'start', user_id: 1 }));
                    };
                    
                    websocket.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    };
                    
                    websocket.onerror = (error) => {
                        console.error('WebSocket 에러:', error);
                        status.textContent = 'WebSocket 연결 실패';
                        status.className = 'status-text error';
                    };
                    
                    websocket.onclose = () => {
                        console.log('WebSocket 종료');
                    };
                    
                    // 마이크 접근
                    realtimeStream = await navigator.mediaDevices.getUserMedia({
                        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
                    });
                    
                    realtimeAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    const source = realtimeAudioContext.createMediaStreamSource(realtimeStream);
                    realtimeProcessor = realtimeAudioContext.createScriptProcessor(4096, 1, 1);
                    
                    realtimeProcessor.onaudioprocess = (e) => {
                        if (!isSpeechRecording || !websocket || websocket.readyState !== WebSocket.OPEN) return;
                        
                        const inputData = e.inputBuffer.getChannelData(0);
                        
                        // 볼륨 표시
                        let sum = 0;
                        for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
                        document.getElementById('volumeLevel').style.width = Math.min(100, Math.sqrt(sum / inputData.length) * 300) + '%';
                        
                        // Float32 -> Int16 변환
                        const pcmData = new Int16Array(inputData.length);
                        for (let i = 0; i < inputData.length; i++) {
                            const s = Math.max(-1, Math.min(1, inputData[i]));
                            pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                        }
                        
                        // Base64 인코딩 후 전송
                        const base64 = arrayBufferToBase64(pcmData.buffer);
                        websocket.send(JSON.stringify({ type: 'audio', data: base64 }));
                    };
                    
                    source.connect(realtimeProcessor);
                    realtimeProcessor.connect(realtimeAudioContext.destination);
                    
                    isSpeechRecording = true;
                    seconds = 0;
                    btn.textContent = '중지';
                    btn.classList.add('recording');
                    status.textContent = '🎤 듣는 중...';
                    status.className = 'status-text';
                    
                    // 타이머
                    timerInterval = setInterval(() => {
                        seconds++;
                        document.getElementById('timer').textContent = Math.floor(seconds / 60) + ':' + (seconds % 60).toString().padStart(2, '0');
                    }, 1000);
                    
                } catch (e) {
                    status.textContent = '마이크 오류: ' + e.message;
                    status.className = 'status-text error';
                }
            } else {
                // 중지
                clearInterval(timerInterval);
                
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({ type: 'stop' }));
                }
                
                if (realtimeProcessor) realtimeProcessor.disconnect();
                if (realtimeStream) realtimeStream.getTracks().forEach(t => t.stop());
                if (realtimeAudioContext) realtimeAudioContext.close();
                
                isSpeechRecording = false;
                btn.textContent = '녹음';
                btn.classList.remove('recording');
                status.textContent = '중지됨';
                document.getElementById('volumeLevel').style.width = '0%';
            }
        }
        
        function handleWebSocketMessage(data) {
            const status = document.getElementById('speechStatus');
            
            switch (data.type) {
                case 'status':
                    const statusText = {
                        'listening': '🎤 듣는 중...',
                        'speaking': '🗣️ 말하는 중...',
                        'silence': '⏸️ 침묵 감지 (2초 후 저장)',
                        'stopped': '⏹️ 종료'
                    };
                    status.textContent = statusText[data.status] || data.status;
                    status.className = 'status-text';
                    if (data.status === 'speaking') status.classList.add('speaking');
                    if (data.status === 'silence') status.classList.add('silence');
                    break;
                    
                case 'result':
                    // 실시간 결과 표시
                    addRealtimeResult(data.text, data.saved);
                    // 대화 기록에 추가
                    if (data.saved) {
                        addHistory(data.text, 'speech');
                    }
                    status.textContent = '🎤 듣는 중...';
                    status.className = 'status-text';
                    break;
                    
                case 'error':
                    status.textContent = '오류: ' + data.message;
                    status.className = 'status-text error';
                    break;
            }
        }
        
        function addRealtimeResult(text, saved) {
            const container = document.getElementById('realtimeResults');
            
            // 첫 결과면 placeholder 제거
            const placeholder = container.querySelector('p');
            if (placeholder) placeholder.remove();
            
            const item = document.createElement('div');
            item.className = 'realtime-item new';
            item.innerHTML = text + (saved ? ' <span style="color:#00ff88">✓ 저장됨</span>' : '');
            container.insertBefore(item, container.firstChild);
            
            // 최대 10개 유지
            while (container.children.length > 10) {
                container.removeChild(container.lastChild);
            }
        }
        
        function arrayBufferToBase64(buffer) {
            const bytes = new Uint8Array(buffer);
            let binary = '';
            for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            return btoa(binary);
        }
        
        // ================================================================
        // 배치 음성 인식 (기존 방식)
        // ================================================================
        
        async function toggleBatchSpeech() {
            const btn = document.getElementById('speechBtn');
            const status = document.getElementById('speechStatus');
            
            if (!isSpeechRecording) {
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
                    audioContext = new AudioContext({ sampleRate: 16000 });
                    const source = audioContext.createMediaStreamSource(mediaStream);
                    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                    audioChunks = [];
                    
                    scriptProcessor.onaudioprocess = (e) => {
                        const inputData = e.inputBuffer.getChannelData(0);
                        audioChunks.push(new Float32Array(inputData));
                        let sum = 0;
                        for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
                        document.getElementById('volumeLevel').style.width = Math.min(100, Math.sqrt(sum / inputData.length) * 300) + '%';
                    };
                    
                    source.connect(scriptProcessor);
                    scriptProcessor.connect(audioContext.destination);
                    
                    isSpeechRecording = true; seconds = 0;
                    btn.textContent = '중지'; btn.classList.add('recording');
                    status.textContent = '녹음 중...';
                    
                    timerInterval = setInterval(() => {
                        seconds++;
                        document.getElementById('timer').textContent = Math.floor(seconds / 60) + ':' + (seconds % 60).toString().padStart(2, '0');
                        if (seconds >= 30) toggleBatchSpeech();
                    }, 1000);
                } catch (e) { status.textContent = '마이크 오류: ' + e.message; }
            } else {
                clearInterval(timerInterval);
                scriptProcessor.disconnect();
                mediaStream.getTracks().forEach(t => t.stop());
                audioContext.close();
                
                isSpeechRecording = false;
                btn.textContent = '녹음'; btn.classList.remove('recording');
                status.textContent = '처리 중...';
                document.getElementById('volumeLevel').style.width = '0%';
                sendAudioPCM();
            }
        }
        
        async function sendAudioPCM() {
            const status = document.getElementById('speechStatus');
            try {
                const totalLength = audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
                const audioData = new Float32Array(totalLength);
                let offset = 0;
                for (const chunk of audioChunks) { audioData.set(chunk, offset); offset += chunk.length; }
                
                const res = await fetch('/speech/predict?session_id=' + sessionId, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/octet-stream' },
                    body: audioData.buffer
                });
                const data = await res.json();
                status.textContent = data.text ? '완료!' : (data.error || '인식 실패');
                document.getElementById('speechResult').textContent = data.text || '-';
                if (data.text) addHistory(data.text, 'speech');
            } catch (e) { status.textContent = '오류: ' + e.message; }
        }
        
        // 카테고리 & 문장
        async function loadCategories() {
            try {
                const res = await fetch('/categories');
                const data = await res.json();
                const container = document.getElementById('categoriesContainer');
                container.innerHTML = '<button class="category-btn active" onclick="loadPhrases(null, this)">전체</button>';
                (data.categories || []).forEach(cat => {
                    const btn = document.createElement('button');
                    btn.className = 'category-btn';
                    btn.textContent = (cat.icon || '') + ' ' + cat.category_name;
                    btn.onclick = () => loadPhrases(cat.category_name, btn);
                    container.appendChild(btn);
                });
            } catch (e) { console.error('Categories error:', e); }
        }
        
        async function loadPhrases(category, btn) {
            if (btn) {
                document.querySelectorAll('.category-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            }
            try {
                const url = category ? '/phrases?category=' + encodeURIComponent(category) : '/phrases';
                const res = await fetch(url);
                const data = await res.json();
                const container = document.getElementById('phrasesContainer');
                container.innerHTML = '';
                if (!data.phrases || data.phrases.length === 0) {
                    container.innerHTML = '<p style="color:#555; text-align:center; padding:20px;">문장이 없습니다</p>';
                    return;
                }
                data.phrases.forEach(p => {
                    const btn = document.createElement('button');
                    btn.className = 'phrase-btn';
                    btn.textContent = p.phrase_text;
                    btn.onclick = () => usePhrase(p.phrase_text);
                    container.appendChild(btn);
                });
            } catch (e) { console.error('Phrases error:', e); }
        }
        
        async function usePhrase(text) {
            document.getElementById('composedText').textContent = text;
            document.getElementById('speechResult').textContent = text;
            addHistory(text, 'phrase');
            try {
                await fetch('/phrase/use', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: 'phrase_text=' + encodeURIComponent(text) + '&session_id=' + sessionId
                });
            } catch (e) {}
        }
        
        function addHistory(text, type) {
            const list = document.getElementById('historyList');
            if (list.querySelector('p')) list.innerHTML = '';
            const item = document.createElement('div');
            item.className = 'history-item';
            const typeLabel = { speech: '음성', sign: '지문자', phrase: '문장' };
            item.innerHTML = '<span>' + text + '</span><span class="history-type ' + type + '">' + (typeLabel[type] || type) + '</span>';
            list.insertBefore(item, list.firstChild);
            while (list.children.length > 20) list.removeChild(list.lastChild);
        }
        
        // 키보드 단축키
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Backspace' && isSignRunning) { e.preventDefault(); backspaceSign(); }
            if (e.key === ' ' && isSignRunning) { e.preventDefault(); addSpaceSign(); }
            if (e.key === 'Escape') { resetSign(); }
            if (e.key === 'Enter') { e.preventDefault(); submitSign(); }
        });
        
        // 초기화
        fetch('/health').then(r => r.json()).then(data => {
            console.log('Health:', data);
            const modelNames = { 'rf': 'RandomForest', 'knn': 'KNN', 'stgcn': 'ST-GCN' };
            document.getElementById('modelBadge').textContent = '모델: ' + (modelNames[data.sign_model_type] || data.sign_model_type);
            if (!data.sign) {
                document.getElementById('signStatus').textContent = '⚠️ 지문자 모델 로드 실패';
                document.getElementById('signStatus').className = 'status-text error';
            }
        });
        
        loadCategories();
        loadPhrases(null);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
