"""
음성 인식 모델 - 디지털 신호처리 직접 구현 버전

[직접 구현한 신호처리 기술]
1. Pre-emphasis 필터 (1차 고역통과 필터)
2. 에너지 기반 VAD (Voice Activity Detection)
3. Zero Crossing Rate 기반 VAD
4. 스펙트럼 서브트랙션 노이즈 제거
5. 위너 필터 (Wiener Filter) 노이즈 제거
6. 대역통과 필터 (Bandpass Filter) - 음성 대역 300Hz~3400Hz
7. 이동평균 필터 (Moving Average Filter)
"""

import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os


# ================================================================
# 직접 구현한 디지털 신호처리 클래스
# ================================================================

class DigitalSignalProcessor:
    """
    디지털 신호처리 모듈 - 모든 필터 직접 구현
    
    구현 원리:
    - Pre-emphasis: y[n] = x[n] - α * x[n-1]
    - VAD: 프레임별 에너지와 ZCR 계산하여 음성 구간 판별
    - 스펙트럼 서브트랙션: |Y(f)|² = |X(f)|² - |N(f)|²
    - 위너 필터: H(f) = |S(f)|² / (|S(f)|² + |N(f)|²)
    - 대역통과 필터: FFT → 주파수 마스킹 → IFFT
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.frame_size = int(0.025 * sample_rate)  # 25ms 프레임
        self.hop_size = int(0.010 * sample_rate)    # 10ms 홉
        
        print("  [DSP] 디지털 신호처리 모듈 초기화")
        print(f"       - 샘플레이트: {sample_rate}Hz")
        print(f"       - 프레임 크기: {self.frame_size} samples ({self.frame_size/sample_rate*1000:.1f}ms)")
        print(f"       - 홉 크기: {self.hop_size} samples ({self.hop_size/sample_rate*1000:.1f}ms)")
    
    # ============================================================
    # 1. Pre-emphasis 필터 (1차 고역통과 필터)
    # ============================================================
    def pre_emphasis(self, signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """
        Pre-emphasis 필터 - 고주파 성분 강조
        
        수식: y[n] = x[n] - α * x[n-1]
        
        목적: 
        - 음성 신호의 고주파 성분은 에너지가 낮음
        - 고주파를 강조하여 SNR 향상
        - 일반적으로 α = 0.95 ~ 0.97 사용
        
        Args:
            signal: 입력 신호
            alpha: 필터 계수 (0.95~0.97)
        
        Returns:
            필터링된 신호
        """
        # 1차 차분 필터 직접 구현
        emphasized = np.zeros_like(signal)
        emphasized[0] = signal[0]
        
        for n in range(1, len(signal)):
            emphasized[n] = signal[n] - alpha * signal[n-1]
        
        return emphasized
    
    # ============================================================
    # 2. 에너지 기반 VAD (Voice Activity Detection)
    # ============================================================
    def compute_frame_energy(self, signal: np.ndarray) -> np.ndarray:
        """
        프레임별 에너지 계산
        
        수식: E[m] = Σ x[n]² (프레임 m 내의 모든 샘플)
        
        Args:
            signal: 입력 신호
        
        Returns:
            프레임별 에너지 배열
        """
        num_frames = (len(signal) - self.frame_size) // self.hop_size + 1
        energies = np.zeros(num_frames)
        
        for m in range(num_frames):
            start = m * self.hop_size
            end = start + self.frame_size
            frame = signal[start:end]
            
            # 에너지 = 샘플 제곱의 합
            energies[m] = np.sum(frame ** 2)
        
        return energies
    
    def compute_zero_crossing_rate(self, signal: np.ndarray) -> np.ndarray:
        """
        프레임별 영교차율(ZCR) 계산
        
        수식: ZCR[m] = (1/N) * Σ |sign(x[n]) - sign(x[n-1])| / 2
        
        특징:
        - 유성음(모음): ZCR 낮음 (저주파 성분 많음)
        - 무성음(자음 's', 'f' 등): ZCR 높음 (고주파 성분 많음)
        - 잡음: ZCR 매우 높음
        
        Args:
            signal: 입력 신호
        
        Returns:
            프레임별 ZCR 배열
        """
        num_frames = (len(signal) - self.frame_size) // self.hop_size + 1
        zcr = np.zeros(num_frames)
        
        for m in range(num_frames):
            start = m * self.hop_size
            end = start + self.frame_size
            frame = signal[start:end]
            
            # 부호 변화 횟수 계산
            signs = np.sign(frame)
            sign_changes = np.abs(np.diff(signs))
            zcr[m] = np.sum(sign_changes > 0) / (2 * len(frame))
        
        return zcr
    
    def energy_vad(self, signal: np.ndarray, 
                   energy_threshold_ratio: float = 0.1,
                   zcr_threshold: float = 0.3) -> tuple:
        """
        에너지 + ZCR 기반 VAD (직접 구현)
        
        알고리즘:
        1. 프레임별 에너지 계산
        2. 프레임별 ZCR 계산
        3. 에너지가 임계값 이상이고 ZCR이 임계값 이하인 구간 = 음성
        
        Args:
            signal: 입력 신호
            energy_threshold_ratio: 에너지 임계값 비율 (최대 에너지 대비)
            zcr_threshold: ZCR 임계값
        
        Returns:
            (음성 존재 여부, 음성 구간 시작/끝 인덱스 리스트)
        """
        energies = self.compute_frame_energy(signal)
        zcr = self.compute_zero_crossing_rate(signal)
        
        # 에너지 임계값 계산 (최대 에너지의 일정 비율)
        energy_threshold = np.max(energies) * energy_threshold_ratio
        
        # 음성 프레임 판별
        # 에너지가 높고 ZCR이 너무 높지 않으면 음성
        speech_frames = (energies > energy_threshold) & (zcr < zcr_threshold)
        
        # 연속된 음성 구간 찾기
        speech_regions = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                speech_regions.append((start_frame, i))
                in_speech = False
        
        if in_speech:
            speech_regions.append((start_frame, len(speech_frames)))
        
        has_speech = len(speech_regions) > 0
        
        return has_speech, speech_regions
    
    def extract_speech_segments(self, signal: np.ndarray, 
                                speech_regions: list) -> np.ndarray:
        """
        음성 구간만 추출하여 연결
        
        Args:
            signal: 원본 신호
            speech_regions: (시작 프레임, 끝 프레임) 리스트
        
        Returns:
            음성 구간만 연결된 신호
        """
        if not speech_regions:
            return signal
        
        segments = []
        for start_frame, end_frame in speech_regions:
            start_sample = start_frame * self.hop_size
            end_sample = min(end_frame * self.hop_size + self.frame_size, len(signal))
            segments.append(signal[start_sample:end_sample])
        
        return np.concatenate(segments) if segments else signal
    
    # ============================================================
    # 3. 스펙트럼 서브트랙션 노이즈 제거
    # ============================================================
    def spectral_subtraction(self, signal: np.ndarray, 
                             noise_frames: int = 10,
                             alpha: float = 2.0,
                             beta: float = 0.01) -> np.ndarray:
        """
        스펙트럼 서브트랙션 노이즈 제거 (직접 구현)
        
        원리:
        1. 초기 몇 프레임을 잡음으로 가정하여 잡음 스펙트럼 추정
        2. 전체 신호의 스펙트럼에서 잡음 스펙트럼을 뺌
        3. 음수가 되는 부분은 작은 값으로 대체 (뮤지컬 노이즈 방지)
        
        수식:
        |Y(f)|² = max(|X(f)|² - α|N(f)|², β|X(f)|²)
        
        Args:
            signal: 입력 신호
            noise_frames: 잡음 추정에 사용할 초기 프레임 수
            alpha: 잡음 제거 강도 (over-subtraction factor)
            beta: 스펙트럼 플로어 (musical noise 방지)
        
        Returns:
            잡음이 제거된 신호
        """
        # STFT 파라미터
        n_fft = self.frame_size
        hop = self.hop_size
        
        # 해밍 윈도우 직접 생성
        window = self._hamming_window(n_fft)
        
        # 프레임 분할
        num_frames = (len(signal) - n_fft) // hop + 1
        frames = np.zeros((num_frames, n_fft))
        
        for i in range(num_frames):
            start = i * hop
            frames[i] = signal[start:start + n_fft] * window
        
        # FFT 수행
        spectra = np.fft.rfft(frames, axis=1)
        magnitude = np.abs(spectra)
        phase = np.angle(spectra)
        
        # 잡음 스펙트럼 추정 (초기 프레임의 평균)
        noise_estimate = np.mean(magnitude[:noise_frames], axis=0)
        
        # 스펙트럼 서브트랙션
        magnitude_squared = magnitude ** 2
        noise_squared = noise_estimate ** 2
        
        # |Y|² = max(|X|² - α|N|², β|X|²)
        clean_magnitude_squared = np.maximum(
            magnitude_squared - alpha * noise_squared,
            beta * magnitude_squared
        )
        clean_magnitude = np.sqrt(clean_magnitude_squared)
        
        # IFFT로 시간 영역 복원
        clean_spectra = clean_magnitude * np.exp(1j * phase)
        clean_frames = np.fft.irfft(clean_spectra, axis=1)
        
        # Overlap-Add 재합성
        output_length = (num_frames - 1) * hop + n_fft
        output = np.zeros(output_length)
        
        for i in range(num_frames):
            start = i * hop
            output[start:start + n_fft] += clean_frames[i]
        
        # 정규화
        output = output / np.max(np.abs(output) + 1e-8)
        
        return output[:len(signal)]
    
    # ============================================================
    # 4. 위너 필터 노이즈 제거
    # ============================================================
    def wiener_filter(self, signal: np.ndarray,
                      noise_frames: int = 10) -> np.ndarray:
        """
        위너 필터 노이즈 제거 (직접 구현)
        
        원리:
        - 최소 평균 제곱 오차(MMSE) 기준으로 최적의 필터
        - 신호 대 잡음비(SNR)에 따라 적응적으로 필터링
        
        수식:
        H(f) = |S(f)|² / (|S(f)|² + |N(f)|²)
             = SNR(f) / (SNR(f) + 1)
        
        Y(f) = H(f) * X(f)
        
        Args:
            signal: 입력 신호
            noise_frames: 잡음 추정에 사용할 초기 프레임 수
        
        Returns:
            필터링된 신호
        """
        n_fft = self.frame_size
        hop = self.hop_size
        window = self._hamming_window(n_fft)
        
        # 프레임 분할 및 FFT
        num_frames = (len(signal) - n_fft) // hop + 1
        frames = np.zeros((num_frames, n_fft))
        
        for i in range(num_frames):
            start = i * hop
            frames[i] = signal[start:start + n_fft] * window
        
        spectra = np.fft.rfft(frames, axis=1)
        magnitude = np.abs(spectra)
        phase = np.angle(spectra)
        
        # 잡음 파워 스펙트럼 추정
        noise_power = np.mean(magnitude[:noise_frames] ** 2, axis=0)
        
        # 신호 파워 스펙트럼
        signal_power = magnitude ** 2
        
        # 위너 필터 계수 계산
        # H(f) = max(1 - noise_power/signal_power, 0)
        # 또는 H(f) = signal_power / (signal_power + noise_power)
        epsilon = 1e-8
        wiener_gain = signal_power / (signal_power + noise_power + epsilon)
        
        # 필터 적용
        filtered_magnitude = magnitude * wiener_gain
        filtered_spectra = filtered_magnitude * np.exp(1j * phase)
        
        # IFFT
        filtered_frames = np.fft.irfft(filtered_spectra, axis=1)
        
        # Overlap-Add
        output_length = (num_frames - 1) * hop + n_fft
        output = np.zeros(output_length)
        
        for i in range(num_frames):
            start = i * hop
            output[start:start + n_fft] += filtered_frames[i]
        
        output = output / np.max(np.abs(output) + 1e-8)
        
        return output[:len(signal)]
    
    # ============================================================
    # 5. 대역통과 필터 (Bandpass Filter)
    # ============================================================
    def bandpass_filter(self, signal: np.ndarray,
                        low_freq: float = 300,
                        high_freq: float = 3400) -> np.ndarray:
        """
        대역통과 필터 (직접 구현) - 음성 주파수 대역만 통과
        
        원리:
        1. FFT로 주파수 영역 변환
        2. 원하는 주파수 대역 외에는 0으로 마스킹
        3. IFFT로 시간 영역 복원
        
        음성 대역: 일반적으로 300Hz ~ 3400Hz (전화 품질)
        
        Args:
            signal: 입력 신호
            low_freq: 저역 차단 주파수 (Hz)
            high_freq: 고역 차단 주파수 (Hz)
        
        Returns:
            필터링된 신호
        """
        n = len(signal)
        
        # FFT
        spectrum = np.fft.rfft(signal)
        frequencies = np.fft.rfftfreq(n, 1/self.sample_rate)
        
        # 주파수 마스크 생성 (부드러운 전이를 위해 코사인 테이퍼 사용)
        mask = np.zeros_like(frequencies)
        
        for i, freq in enumerate(frequencies):
            if low_freq <= freq <= high_freq:
                mask[i] = 1.0
            elif freq < low_freq:
                # 저주파 롤오프 (코사인 테이퍼)
                if freq > low_freq * 0.8:
                    mask[i] = 0.5 * (1 + np.cos(np.pi * (low_freq - freq) / (low_freq * 0.2)))
            else:
                # 고주파 롤오프
                if freq < high_freq * 1.2:
                    mask[i] = 0.5 * (1 + np.cos(np.pi * (freq - high_freq) / (high_freq * 0.2)))
        
        # 필터 적용
        filtered_spectrum = spectrum * mask
        
        # IFFT
        filtered_signal = np.fft.irfft(filtered_spectrum, n)
        
        return filtered_signal
    
    # ============================================================
    # 6. 이동평균 필터 (Moving Average Filter)
    # ============================================================
    def moving_average_filter(self, signal: np.ndarray, 
                              window_size: int = 5) -> np.ndarray:
        """
        이동평균 필터 (직접 구현) - 고주파 잡음 제거
        
        수식: y[n] = (1/M) * Σ x[n-k] for k=0 to M-1
        
        특징:
        - 저역통과 필터 역할
        - 구현이 간단하고 계산 효율적
        - 고주파 잡음 제거에 효과적
        
        Args:
            signal: 입력 신호
            window_size: 윈도우 크기 (M)
        
        Returns:
            필터링된 신호
        """
        output = np.zeros_like(signal)
        
        for n in range(len(signal)):
            start = max(0, n - window_size + 1)
            output[n] = np.mean(signal[start:n+1])
        
        return output
    
    # ============================================================
    # 7. 해밍 윈도우 (직접 구현)
    # ============================================================
    def _hamming_window(self, size: int) -> np.ndarray:
        """
        해밍 윈도우 직접 구현
        
        수식: w[n] = 0.54 - 0.46 * cos(2πn / (N-1))
        
        목적:
        - 프레임 경계에서의 불연속성 감소
        - 스펙트럼 누설(spectral leakage) 감소
        
        Args:
            size: 윈도우 크기
        
        Returns:
            해밍 윈도우 배열
        """
        n = np.arange(size)
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (size - 1))
    
    # ============================================================
    # 통합 전처리 파이프라인
    # ============================================================
    def process(self, signal: np.ndarray, 
                apply_preemphasis: bool = True,
                apply_bandpass: bool = True,
                apply_vad: bool = True,
                apply_denoise: bool = True,
                denoise_method: str = "wiener") -> dict:
        """
        통합 신호처리 파이프라인
        
        처리 순서:
        1. Pre-emphasis (고주파 강조)
        2. 대역통과 필터 (음성 대역 추출)
        3. VAD (음성 구간 검출)
        4. 노이즈 제거 (스펙트럼 서브트랙션 또는 위너 필터)
        
        Args:
            signal: 입력 신호
            apply_preemphasis: Pre-emphasis 적용 여부
            apply_bandpass: 대역통과 필터 적용 여부
            apply_vad: VAD 적용 여부
            apply_denoise: 노이즈 제거 적용 여부
            denoise_method: "spectral" 또는 "wiener"
        
        Returns:
            처리 결과 딕셔너리
        """
        processed = signal.copy()
        has_speech = True
        speech_regions = []
        
        # 1. Pre-emphasis
        if apply_preemphasis:
            processed = self.pre_emphasis(processed, alpha=0.97)
            print("    [DSP] Pre-emphasis 적용 (α=0.97)")
        
        # 2. 대역통과 필터
        if apply_bandpass:
            processed = self.bandpass_filter(processed, low_freq=300, high_freq=3400)
            print("    [DSP] 대역통과 필터 적용 (300-3400Hz)")
        
        # 3. VAD
        if apply_vad:
            has_speech, speech_regions = self.energy_vad(processed)
            if has_speech:
                processed = self.extract_speech_segments(processed, speech_regions)
                print(f"    [DSP] VAD: {len(speech_regions)}개 음성 구간 검출")
            else:
                print("    [DSP] VAD: 음성 없음")
        
        # 4. 노이즈 제거
        if apply_denoise and has_speech and len(processed) > self.frame_size * 2:
            if denoise_method == "spectral":
                processed = self.spectral_subtraction(processed)
                print("    [DSP] 스펙트럼 서브트랙션 노이즈 제거")
            else:
                processed = self.wiener_filter(processed)
                print("    [DSP] 위너 필터 노이즈 제거")
        
        return {
            "signal": processed,
            "has_speech": has_speech,
            "speech_regions": speech_regions,
            "original_length": len(signal),
            "processed_length": len(processed)
        }


# ================================================================
# 음성 인식 클래스
# ================================================================

class SpeechRecognizer:
    """
    음성 인식 모델
    
    특징:
    - 직접 구현한 디지털 신호처리 모듈 사용
    - Whisper 모델로 음성→텍스트 변환
    """
    
    def __init__(self, model_size="medium"):
        self.sample_rate = 16000
        self.device = None
        self.model = None
        self.processor = None
        
        # 직접 구현한 신호처리 모듈
        self.dsp = DigitalSignalProcessor(sample_rate=self.sample_rate)
        
        self._load_whisper(model_size)
        print("  ✅ 음성 인식 모델 초기화 완료")
    
    def _load_whisper(self, model_size):
        """Whisper 모델 로드"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        possible_paths = [
            os.path.join(script_dir, f"whisper-{model_size}"),
            os.path.join(script_dir, "weights", f"whisper-{model_size}"),
            os.path.join(script_dir, "models", f"whisper-{model_size}"),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f"Whisper 모델({model_size})을 찾을 수 없습니다.\n"
                f"다음 명령어로 다운로드하세요: python scripts/download_whisper.py"
            )
        
        print(f"  📂 Whisper 모델 경로: {model_path}")
        
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("  🖥️ Apple Silicon (MPS) 사용")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("  🖥️ CUDA GPU 사용")
        else:
            self.device = torch.device("cpu")
            print("  🖥️ CPU 사용")
        
        self.model.to(self.device)
        self.model.eval()
        print("  ✅ Whisper 로드 완료")
    
    def is_valid_text(self, text: str) -> bool:
        """텍스트 유효성 검사 (환각 필터링)"""
        text = text.strip()
        
        if not text or len(text) < 2:
            return False
        
        bad_patterns = [
            "MBC", "KBS", "SBS", "YTN", "JTBC", "뉴스", "기자입니다", "기자",
            "시청해", "구독", "좋아요", "알림", "자막", "번역",
            "Translated", "Subtitles", "Thank you", "Subscribe",
            "ご視聴", "ありがとう", "...", "…",
        ]
        
        for pattern in bad_patterns:
            if pattern in text:
                print(f"    [필터링: {text}]")
                return False
        
        # 반복 단어 검사
        words = text.split()
        if len(words) >= 3:
            for word in set(words):
                if words.count(word) >= 3 and len(word) > 1:
                    print(f"    [반복 필터링: {text}]")
                    return False
        
        return True
    
    def transcribe(self, audio: np.ndarray) -> str:
        """Whisper로 음성 인식"""
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)
        
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="korean",
            task="transcribe"
        )
        
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=128,
                no_repeat_ngram_size=3,
                condition_on_prev_tokens=False,
            )
        
        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return text.strip()
    
    def predict(self, audio: np.ndarray, use_dsp: bool = False) -> dict:
        """
        메인 예측 함수
        
        처리 순서:
        1. 디지털 신호처리 (직접 구현) - 선택적
        2. Whisper 음성 인식
        3. 텍스트 후처리
        
        Args:
            audio: 입력 오디오
            use_dsp: 신호처리 적용 여부 (기본 False - Whisper가 자체 처리)
        """
        duration = len(audio) / self.sample_rate
        print(f"  🎤 입력 오디오: {duration:.1f}초, {len(audio)} samples")
        
        if duration < 0.5:
            return {"text": "", "has_speech": False, "processed": False}
        
        # ========================================
        # 신호처리 (선택적 적용)
        # ========================================
        # 참고: Whisper는 자체적으로 강력한 전처리를 수행하므로
        # 추가 신호처리가 오히려 성능을 저하시킬 수 있음.
        # 신호처리 기술 시연이 필요할 때만 use_dsp=True 사용.
        
        dsp_result = None
        
        if use_dsp:
            print("  📊 신호처리 시작...")
            dsp_result = self.dsp.process(
                audio,
                apply_preemphasis=True,
                apply_bandpass=False,  # 대역통과는 Whisper가 처리
                apply_vad=True,
                apply_denoise=False,   # 노이즈 제거는 가볍게
                denoise_method="wiener"
            )
            
            if not dsp_result["has_speech"]:
                print("  ⚠️ 음성 없음 (VAD)")
                return {"text": "", "has_speech": False, "processed": True}
            
            processed_audio = dsp_result["signal"]
            print(f"  📊 신호처리 완료: {dsp_result['original_length']} → {dsp_result['processed_length']} samples")
        else:
            # 기본: 가벼운 전처리만 (Pre-emphasis)
            processed_audio = self.dsp.pre_emphasis(audio, alpha=0.97)
            print("  📊 Pre-emphasis만 적용")
        
        # ========================================
        # Whisper 음성 인식
        # ========================================
        print("  🔊 Whisper 인식 중...")
        text = self.transcribe(processed_audio)
        print(f"  📝 인식 결과: {text}")
        
        if not self.is_valid_text(text):
            return {"text": "", "has_speech": True, "processed": True}
        
        print(f"  ✅ 최종 결과: {text}")
        
        result = {
            "text": text, 
            "has_speech": True, 
            "processed": True,
        }
        
        if dsp_result:
            result["dsp_info"] = {
                "original_length": dsp_result["original_length"],
                "processed_length": dsp_result["processed_length"],
                "speech_regions": len(dsp_result["speech_regions"])
            }
        
        return result


# ================================================================
# 테스트
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("음성 인식 모델 테스트 - 디지털 신호처리 직접 구현")
    print("=" * 60)
    
    # DSP 모듈 테스트
    print("\n[1] 디지털 신호처리 모듈 테스트")
    dsp = DigitalSignalProcessor(sample_rate=16000)
    
    # 테스트 신호 생성 (1초, 440Hz 사인파 + 잡음)
    t = np.linspace(0, 1, 16000)
    test_signal = np.sin(2 * np.pi * 440 * t) + 0.3 * np.random.randn(16000)
    
    print("\n  [Pre-emphasis 테스트]")
    emphasized = dsp.pre_emphasis(test_signal)
    print(f"    입력 분산: {np.var(test_signal):.4f}")
    print(f"    출력 분산: {np.var(emphasized):.4f}")
    
    print("\n  [대역통과 필터 테스트]")
    bandpassed = dsp.bandpass_filter(test_signal, 300, 3400)
    print(f"    입력 분산: {np.var(test_signal):.4f}")
    print(f"    출력 분산: {np.var(bandpassed):.4f}")
    
    print("\n  [VAD 테스트]")
    has_speech, regions = dsp.energy_vad(test_signal)
    print(f"    음성 존재: {has_speech}")
    print(f"    음성 구간: {len(regions)}개")
    
    print("\n  [위너 필터 테스트]")
    denoised = dsp.wiener_filter(test_signal)
    print(f"    입력 분산: {np.var(test_signal):.4f}")
    print(f"    출력 분산: {np.var(denoised):.4f}")
    
    print("\n  [통합 파이프라인 테스트]")
    result = dsp.process(test_signal)
    print(f"    원본 길이: {result['original_length']}")
    print(f"    처리 후 길이: {result['processed_length']}")
    print(f"    음성 존재: {result['has_speech']}")
    
    print("\n" + "=" * 60)
    print("✅ 모든 디지털 신호처리 모듈 테스트 완료!")
    print("=" * 60)