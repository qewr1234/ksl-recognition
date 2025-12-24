"""
지문자 인식 모델 - RandomForest (자동 학습)

모델 우선순위:
1. ST-GCN (stgcn_best.pth)
2. RandomForest (CSV에서 자동 학습, ~92% 정확도)
3. KNN 폴백 (~85% 정확도)
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque, Counter
import time
import warnings
import pickle

warnings.filterwarnings("ignore")

# sklearn 사용 가능 여부
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("  [WARN] sklearn not installed. pip install scikit-learn")


# ============ ST-GCN 모델 정의 ============

class HandGraph:
    num_nodes = 21
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    
    @classmethod
    def get_normalized_adjacency(cls):
        A = np.zeros((cls.num_nodes, cls.num_nodes), dtype=np.float32)
        for i, j in cls.edges:
            A[i, j] = 1
            A[j, i] = 1
        A += np.eye(cls.num_nodes, dtype=np.float32)
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
        return torch.tensor(D_inv_sqrt @ A @ D_inv_sqrt, dtype=torch.float32)


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = A
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', x, self.A.to(x.device))
        return F.relu(self.bn(self.conv(x)))


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1), 
                              padding=((kernel_size - 1) // 2, 0))
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.sgc = SpatialGraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels)
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        return F.relu(self.dropout(self.tcn(self.sgc(x))) + self.residual(x))


class STGCN(nn.Module):
    def __init__(self, num_classes=31):
        super().__init__()
        A = HandGraph.get_normalized_adjacency()
        self.register_buffer('A', A)
        self.input_bn = nn.BatchNorm2d(3)
        self.layers = nn.ModuleList([
            STGCNBlock(3, 64, A), STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A), STGCNBlock(128, 128, A),
            STGCNBlock(128, 256, A),
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.input_bn(x.permute(0, 3, 1, 2).contiguous())
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.gap(x).view(x.size(0), -1))


# ============ 지문자 인식기 ============

class FingerSpellRecognizer:
    """RandomForest/KNN 기반 지문자 인식 + 완전한 한글 조합"""
    
    GESTURE = {
        0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ',
        8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ',
        14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ',
        20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ',
        24: 'ㅐ', 25: 'ㅔ', 26: 'ㅚ', 27: 'ㅟ', 28: 'ㅢ', 29: 'ㅒ', 30: 'ㅖ',
    }
    
    CHOSUNG_LABELS = set(range(0, 14))
    JUNGSUNG_LABELS = set(range(14, 31))
    
    LABEL_TO_CHOSUNG = {0:0, 1:2, 2:3, 3:5, 4:6, 5:7, 6:9, 7:11, 8:12, 9:14, 10:15, 11:16, 12:17, 13:18}
    LABEL_TO_JUNGSUNG = {14:0, 15:2, 16:4, 17:6, 18:8, 19:12, 20:13, 21:17, 22:18, 23:20, 24:1, 25:5, 26:11, 27:16, 28:19, 29:3, 30:7}
    LABEL_TO_JONGSUNG = {0:1, 1:4, 2:7, 3:8, 4:16, 5:17, 6:19, 7:21, 8:22, 9:23, 10:24, 11:25, 12:26, 13:27}
    
    def __init__(self):
        self.model = None
        self.model_type = None  # 'stgcn', 'rf', 'knn'
        self.rf_model = None
        self.knn = None
        self.hands = None
        self.device = None
        
        # ST-GCN용
        self.frame_buffer = deque(maxlen=30)
        
        # 인식 파라미터
        self.prediction_buffer = deque(maxlen=15)
        self.last_confirmed = None
        self.last_confirm_time = 0
        self.confirm_threshold = 0.60  # 60%로 낮춤
        self.cooldown = 1.2  # 1.2초
        
        # 한글 조합
        self.state = "EMPTY"
        self.current_cho = None
        self.current_jung = None
        self.current_jong = None
        self.composed_text = ""
        
        # 마지막 완성된 글자 (DB 저장용)
        self.last_completed_char = ""
        
        self._load_model()
        self._load_mediapipe()
    
    def _load_model(self):
        """모델 로드 (ST-GCN → RandomForest → KNN)"""
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"  [INFO] Device: {self.device}")
        
        # 프로젝트 루트 경로 계산
        current_dir = Path(__file__).parent  # models/
        project_root = current_dir.parent     # ksl-recognition/
        
        # 1. ST-GCN 시도
        stgcn_paths = [
            project_root / "weights" / "stgcn_best.pth",
            project_root / "models" / "stgcn_best.pth",
            current_dir / "stgcn_best.pth",
        ]
        for p in stgcn_paths:
            if p.exists():
                try:
                    self.model = STGCN(num_classes=31).to(self.device)
                    self.model.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
                    self.model.eval()
                    self.model_type = 'stgcn'
                    print(f"  [INFO] ST-GCN loaded: {p}")
                    return
                except Exception as e:
                    print(f"  [WARN] ST-GCN 실패: {e}")
        
        # 2. CSV 데이터 찾기
        csv_paths = [
            project_root / "data" / "fingerspell_data.csv",
            current_dir / "fingerspell_data.csv",
        ]
        
        data_path = None
        for p in csv_paths:
            if p.exists():
                data_path = p
                break
        
        if data_path is None:
            print("  [ERROR] 데이터 파일을 찾을 수 없습니다!")
            return
        
        # 데이터 로드
        try:
            data = np.genfromtxt(str(data_path), delimiter=',')
            data = data[~np.isnan(data).any(axis=1)]
            X = data[:, :-1].astype(np.float32)
            y = data[:, -1].astype(np.int32)
            print(f"  [INFO] Data loaded: {len(X)} samples")
        except Exception as e:
            print(f"  [ERROR] 데이터 로드 실패: {e}")
            return
        
        # 3. RandomForest 시도 (캐시 또는 학습)
        rf_cache = data_path.parent / "rf_model.pkl"
        
        if SKLEARN_AVAILABLE:
            if rf_cache.exists():
                try:
                    with open(rf_cache, 'rb') as f:
                        self.rf_model = pickle.load(f)
                    self.model_type = 'rf'
                    print(f"  [INFO] RandomForest loaded from cache")
                    return
                except:
                    pass
            
            # 새로 학습 (1-2초)
            try:
                print("  [INFO] Training RandomForest...")
                self.rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    n_jobs=-1,
                    random_state=42
                )
                self.rf_model.fit(X, y)
                self.model_type = 'rf'
                
                # 캐시 저장
                try:
                    with open(rf_cache, 'wb') as f:
                        pickle.dump(self.rf_model, f)
                    print(f"  [INFO] RandomForest trained & cached")
                except:
                    print(f"  [INFO] RandomForest trained (cache failed)")
                return
            except Exception as e:
                print(f"  [WARN] RandomForest 실패: {e}")
        
        # 4. KNN 폴백
        try:
            self.knn = cv2.ml.KNearest_create()
            self.knn.train(X, cv2.ml.ROW_SAMPLE, y.astype(np.float32))
            self.model_type = 'knn'
            print(f"  [INFO] KNN loaded: {len(X)} samples")
        except Exception as e:
            print(f"  [ERROR] KNN 실패: {e}")
    
    def _load_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        print("  [INFO] MediaPipe ready")
    
    def _get_landmarks(self, image):
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
            points = [{"x": lm.x, "y": lm.y} for lm in hand.landmark]
            return coords, [{"hand": "right", "points": points}]
        return None, []
    
    def _extract_angles(self, landmarks):
        joint = landmarks
        v1 = joint[[0,1,2,3, 0,5,6,7, 0,9,10,11, 0,13,14,15, 0,17,18,19], :]
        v2 = joint[[1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20], :]
        v = v2 - v1
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1
        v = v / norms
        idx1 = [0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18]
        idx2 = [1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19]
        dot = np.clip(np.einsum('ij,ij->i', v[idx1], v[idx2]), -1.0, 1.0)
        return np.degrees(np.arccos(dot)).astype(np.float32)
    
    def _predict_rf(self, landmarks):
        """RandomForest 예측"""
        angles = self._extract_angles(landmarks)
        pred = self.rf_model.predict([angles])[0]
        proba = self.rf_model.predict_proba([angles])[0]
        confidence = proba[pred]
        return int(pred), float(confidence), "rf"
    
    def _predict_knn(self, landmarks):
        """KNN 예측"""
        angles = self._extract_angles(landmarks)
        ret, results, neighbours, dist = self.knn.findNearest(np.array([angles]), 5)
        pred_idx = int(results[0][0])
        confidence = max(0, min(1, 1 - (np.mean(dist[0]) / 3000)))
        return pred_idx, confidence, "knn"
    
    def _predict_stgcn(self, landmarks):
        """ST-GCN 예측"""
        self.frame_buffer.append(landmarks)
        if len(self.frame_buffer) < 30:
            return None, 0, f"버퍼링 {len(self.frame_buffer)}/30"
        
        seq = np.array(list(self.frame_buffer))
        seq = (seq - seq.mean()) / (seq.std() + 1e-6)
        
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            probs = F.softmax(self.model(x), dim=1)
            conf, pred = probs.max(1)
            return pred.item(), conf.item(), "stgcn"
    
    # ============ 한글 조합 ============
    
    def _make_syllable(self, cho, jung, jong=None):
        if cho is None or jung is None:
            return None
        cho_idx = self.LABEL_TO_CHOSUNG.get(cho)
        jung_idx = self.LABEL_TO_JUNGSUNG.get(jung)
        jong_idx = self.LABEL_TO_JONGSUNG.get(jong, 0) if jong else 0
        if cho_idx is None or jung_idx is None:
            return None
        return chr(0xAC00 + cho_idx * 588 + jung_idx * 28 + jong_idx)
    
    def _get_current_display(self):
        if self.state == "EMPTY":
            return ""
        elif self.state == "CHO":
            return self.GESTURE.get(self.current_cho, "")
        elif self.state == "CHO_JUNG":
            return self._make_syllable(self.current_cho, self.current_jung) or ""
        elif self.state == "CHO_JUNG_JONG":
            return self._make_syllable(self.current_cho, self.current_jung, self.current_jong) or ""
        return ""
    
    def _process_input(self, label):
        """한글 조합 처리 - 완성된 글자 반환"""
        char = self.GESTURE.get(label, "?")
        is_consonant = label in self.CHOSUNG_LABELS
        is_vowel = label in self.JUNGSUNG_LABELS
        completed = ""  # 완성된 글자
        
        if self.state == "EMPTY":
            if is_consonant:
                self.current_cho = label
                self.state = "CHO"
            elif is_vowel:
                self.composed_text += char
                completed = char
        
        elif self.state == "CHO":
            if is_consonant:
                # 이전 초성 낱자 확정
                prev = self.GESTURE.get(self.current_cho, "")
                self.composed_text += prev
                completed = prev
                self.current_cho = label
            elif is_vowel:
                self.current_jung = label
                self.state = "CHO_JUNG"
        
        elif self.state == "CHO_JUNG":
            if is_consonant:
                # 종성 추가
                self.current_jong = label
                self.state = "CHO_JUNG_JONG"
            elif is_vowel:
                # 현재 음절 확정 + 모음 낱자
                syllable = self._make_syllable(self.current_cho, self.current_jung)
                if syllable:
                    self.composed_text += syllable
                    completed = syllable
                self.composed_text += char
                completed += char
                self._reset_current()
        
        elif self.state == "CHO_JUNG_JONG":
            if is_consonant:
                # 현재 음절 확정 + 새 초성
                syllable = self._make_syllable(self.current_cho, self.current_jung, self.current_jong)
                if syllable:
                    self.composed_text += syllable
                    completed = syllable
                self.current_cho = label
                self.current_jung = None
                self.current_jong = None
                self.state = "CHO"
            elif is_vowel:
                # ★ 종성 → 다음 초성으로 이동 ★
                syllable = self._make_syllable(self.current_cho, self.current_jung)
                if syllable:
                    self.composed_text += syllable
                    completed = syllable
                self.current_cho = self.current_jong
                self.current_jung = label
                self.current_jong = None
                self.state = "CHO_JUNG"
        
        self.last_completed_char = completed
        return completed
    
    def _reset_current(self):
        self.state = "EMPTY"
        self.current_cho = None
        self.current_jung = None
        self.current_jong = None
    
    # ============ 메인 API ============
    
    def predict(self, video_bytes):
        nparr = np.frombuffer(video_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return self._result("", "image error", "none", [])
        
        landmarks, landmarks_draw = self._get_landmarks(image)
        
        if landmarks is None:
            display = self._get_current_display()
            return self._result("", "no hand", "none", [],
                               current_char=display,
                               composed=self.composed_text + display)
        
        now = time.time()
        
        # 쿨다운 체크 (쿨다운 끝나면 last_confirmed 리셋!)
        if now - self.last_confirm_time < self.cooldown:
            remaining = self.cooldown - (now - self.last_confirm_time)
            display = self._get_current_display()
            return self._result("", f"대기 {remaining:.1f}s", "right", landmarks_draw,
                               current_char=display,
                               composed=self.composed_text + display)
        else:
            # ★ 쿨다운 끝나면 같은 글자도 다시 인식 가능 ★
            self.last_confirmed = None
        
        # 예측
        if self.model_type == 'stgcn' and self.model:
            pred_idx, confidence, method = self._predict_stgcn(landmarks)
            if pred_idx is None:
                display = self._get_current_display()
                return self._result("", method, "right", landmarks_draw,
                                   current_char=display,
                                   composed=self.composed_text + display)
        elif self.model_type == 'rf' and self.rf_model:
            pred_idx, confidence, method = self._predict_rf(landmarks)
        elif self.knn:
            pred_idx, confidence, method = self._predict_knn(landmarks)
        else:
            return self._result("", "no model", "right", landmarks_draw)
        
        # 버퍼에 추가
        self.prediction_buffer.append(pred_idx)
        predicted_char = self.GESTURE.get(pred_idx, "?")
        display = self._get_current_display()
        
        # 다수결 확인
        if len(self.prediction_buffer) >= 10:
            counter = Counter(self.prediction_buffer)
            most_common, count = counter.most_common(1)[0]
            ratio = count / len(self.prediction_buffer)
            
            if ratio >= self.confirm_threshold:
                self.last_confirmed = most_common
                self.last_confirm_time = now
                self.prediction_buffer.clear()
                if self.model_type == 'stgcn':
                    self.frame_buffer.clear()
                
                # 한글 조합
                completed = self._process_input(most_common)
                confirmed = self.GESTURE.get(most_common, "?")
                display = self._get_current_display()
                
                print(f"  [인식] {confirmed} ({method}, {ratio:.0%})")
                
                return self._result(
                    completed, "recognized", "right", landmarks_draw,
                    confidence=ratio, current_char=display,
                    composed=self.composed_text + display,
                    completed_char=completed
                )
        
        status = f"인식 중: {predicted_char} ({len(self.prediction_buffer)}/{self.prediction_buffer.maxlen})"
        return self._result("", status, "right", landmarks_draw,
                           confidence=confidence,
                           current_char=display or predicted_char,
                           composed=self.composed_text + display)
    
    def _result(self, text, status, hands, landmarks, confidence=0, 
                current_char="", composed="", completed_char=""):
        return {
            "text": text,
            "status": status,
            "hands": hands,
            "landmarks": landmarks,
            "confidence": confidence,
            "current_char": current_char,
            "composed_text": composed,
            "completed_char": completed_char,  # DB 저장용
            "buffer": len(self.prediction_buffer),
            "buffer_max": self.prediction_buffer.maxlen,
            "model_type": self.model_type or "none"
        }
    
    def reset(self):
        self.prediction_buffer.clear()
        self.frame_buffer.clear()
        self._reset_current()
        self.composed_text = ""
        self.last_confirmed = None
        self.last_confirm_time = 0
        self.last_completed_char = ""
    
    def backspace(self):
        if self.state != "EMPTY":
            if self.state == "CHO_JUNG_JONG":
                self.current_jong = None
                self.state = "CHO_JUNG"
            elif self.state == "CHO_JUNG":
                self.current_jung = None
                self.state = "CHO"
            elif self.state == "CHO":
                self._reset_current()
        elif self.composed_text:
            self.composed_text = self.composed_text[:-1]
        return self.composed_text + self._get_current_display()
    
    def add_space(self):
        display = self._get_current_display()
        if display:
            self.composed_text += display
        self._reset_current()
        self.composed_text += " "
        return self.composed_text
    
    def get_composed_text(self):
        """현재까지 완성된 전체 텍스트"""
        return self.composed_text + self._get_current_display()


# 호환성
class SignRecognizer(FingerSpellRecognizer):
    pass


if __name__ == "__main__":
    print("=" * 50)
    print("[TEST] Finger Spell Recognition")
    print("=" * 50)
    
    r = FingerSpellRecognizer()
    print(f"\nModel type: {r.model_type}")
    print(f"Device: {r.device}")
    
    # 한글 조합 테스트
    print("\n[한글 조합 테스트]")
    print("ㅇ + ㅏ + ㅇ + ㅇ 입력:")
    r._process_input(7)   # ㅇ
    print(f"  → {r.get_composed_text()}")
    r._process_input(14)  # ㅏ
    print(f"  → {r.get_composed_text()}")
    r._process_input(7)   # ㅇ (종성)
    print(f"  → {r.get_composed_text()}")
    r._process_input(7)   # ㅇ (새 초성)
    print(f"  → {r.get_composed_text()}")