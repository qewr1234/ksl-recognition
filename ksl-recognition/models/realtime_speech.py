"""
실시간 음성 인식 모듈
"""

import numpy as np
import asyncio
import time
from collections import deque
from typing import Optional, Callable
from dataclasses import dataclass

from speech_model import DigitalSignalProcessor, SpeechRecognizer


@dataclass
class RealtimeConfig:
    sample_rate: int = 16000
    chunk_duration: float = 0.1      # 100ms 청크
    silence_threshold: float = 2.0   # 2초 침묵 시 문장 완료
    min_speech_duration: float = 0.5 # 최소 0.5초 이상
    energy_threshold: float = 0.01   # 에너지 임계값
    max_buffer_duration: float = 30.0


class EnergyVAD:
    """에너지 기반 VAD"""
    def __init__(self, sample_rate=16000, energy_threshold=0.01, smoothing_window=5):
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
    def __init__(self, config=None):
        self.config = config or RealtimeConfig()
        self.audio_buffer = []
        self.vad = EnergyVAD(energy_threshold=self.config.energy_threshold)
        self.last_speech_time = None
        self.speech_started = False
        self.dsp = DigitalSignalProcessor(sample_rate=self.config.sample_rate)
        self.recognizer = None  # Lazy loading
        
        # 콜백
        self.on_final_result = None
        self.on_status_change = None
        self.is_running = False
    
    def _ensure_recognizer(self):
        if self.recognizer is None:
            self.recognizer = SpeechRecognizer(model_size="medium")
    
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
        self.vad.reset()
    
    def process_chunk(self, audio_chunk):
        """오디오 청크 처리 - 2초 침묵 시 자동 인식"""
        if not self.is_running:
            return None
        
        current_time = time.time()
        self.audio_buffer.append(audio_chunk)
        
        # VAD로 음성 감지
        is_speech = self.vad.is_speech(audio_chunk)
        
        if is_speech:
            self.last_speech_time = current_time
            if not self.speech_started:
                self.speech_started = True
                if self.on_status_change:
                    self.on_status_change("speaking")
        else:
            if self.speech_started:
                if self.on_status_change:
                    self.on_status_change("silence")
            
            # 2초 침묵 감지 → 인식 수행
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
        self.vad.reset()
        if self.on_status_change:
            self.on_status_change("listening")
        return result
    
    def _process_buffer(self):
        if not self.audio_buffer:
            return None
        
        audio = np.concatenate(self.audio_buffer)
        duration = len(audio) / self.config.sample_rate
        
        if duration < self.config.min_speech_duration:
            return None
        
        self._ensure_recognizer()
        processed_audio = self.dsp.pre_emphasis(audio, alpha=0.97)
        result = self.recognizer.predict(processed_audio, use_dsp=False)
        
        if result["text"]:
            if self.on_final_result:
                self.on_final_result(result["text"])
            return {"text": result["text"], "duration": duration}
        return None


class RealtimeAudioProcessor:
    """WebSocket 오디오 처리기"""
    def __init__(self, recognizer):
        self.recognizer = recognizer
    
    def process_audio_base64(self, audio_base64):
        import base64
        audio_bytes = base64.b64decode(audio_base64)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        return self.recognizer.process_chunk(audio_float)