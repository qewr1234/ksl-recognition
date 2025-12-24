"""
ST-GCN 지문자 데이터 자동 수집기
- 라벨 선택 후 자동으로 0.5초마다 30프레임 시퀀스 저장
- 클래스당 50개 시퀀스 (약 25초)

사용법:
1. python auto_collect.py
2. 라벨 선택 (0~9, a~v)
3. SPACE로 자동 수집 시작
4. 50개 완료되면 자동 중지
5. 다음 라벨 선택 후 반복
6. 's'로 저장, 'q'로 종료
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import time

# 라벨 매핑
LABEL_MAP = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'a': 10, 'b': 11, 'c': 12, 'd': 13,
    'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18,
    'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23,
    'o': 24, 'p': 25, 'q': 26, 'r': 27, 't': 28,
    'u': 29, 'v': 30,
}

GESTURE_NAME = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ',
    8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ',
    14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ',
    20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ',
    24: 'ㅐ', 25: 'ㅔ', 26: 'ㅚ', 27: 'ㅟ', 28: 'ㅢ', 29: 'ㅒ', 30: 'ㅖ',
}


class AutoDataCollector:
    def __init__(self, output_dir="collected_data", samples_per_class=50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.current_label = None
        self.frame_buffer = []
        self.collected_data = []
        
        # 설정
        self.seq_length = 30
        self.samples_per_class = samples_per_class
        self.save_interval = 0.5  # 0.5초마다 저장
        
        # 상태
        self.is_auto_collecting = False
        self.last_save_time = 0
        self.current_class_count = 0
        
        # 클래스별 수집 현황
        self.class_counts = {i: 0 for i in range(31)}
    
    def extract_landmarks(self, image):
        """손 좌표 추출 (21 x 3)"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            coords = []
            for lm in hand.landmark:
                coords.append([lm.x, lm.y, lm.z])
            return np.array(coords, dtype=np.float32), hand
        return None, None
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)  # 높은 FPS
        
        print("=" * 50)
        print("ST-GCN 자동 데이터 수집기")
        print("=" * 50)
        print(f"설정: {self.samples_per_class}개/클래스, {self.save_interval}초 간격")
        print(f"예상 시간: 클래스당 {self.samples_per_class * self.save_interval:.0f}초")
        print("-" * 50)
        print("조작법:")
        print("  0~9, a~v: 라벨 선택")
        print("  SPACE: 자동 수집 시작/중지")
        print("  s: 전체 저장")
        print("  q: 종료")
        print("=" * 50)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            landmarks, hand = self.extract_landmarks(frame)
            
            # 손 그리기
            if hand:
                self.mp_draw.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS)
                
                # 프레임 버퍼에 추가
                if landmarks is not None:
                    self.frame_buffer.append(landmarks)
                    
                    # 버퍼 크기 제한
                    if len(self.frame_buffer) > self.seq_length:
                        self.frame_buffer.pop(0)
                
                # 자동 수집 모드
                if self.is_auto_collecting and self.current_label is not None:
                    current_time = time.time()
                    
                    # 0.5초마다 시퀀스 저장
                    if (current_time - self.last_save_time >= self.save_interval and 
                        len(self.frame_buffer) >= self.seq_length):
                        
                        self._save_sequence()
                        self.last_save_time = current_time
                        
                        # 목표 개수 도달 시 자동 중지
                        if self.current_class_count >= self.samples_per_class:
                            self.is_auto_collecting = False
                            print(f"\n✅ {GESTURE_NAME[self.current_label]} 수집 완료! ({self.samples_per_class}개)")
            
            # UI 표시
            self._draw_ui(frame)
            
            cv2.imshow('Auto Data Collector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self._toggle_auto_collect()
            elif key == ord('s'):
                self._save_all()
            elif chr(key) in LABEL_MAP:
                self._select_label(chr(key))
        
        cap.release()
        cv2.destroyAllWindows()
        self._save_all()
    
    def _select_label(self, key):
        """라벨 선택"""
        self.current_label = LABEL_MAP[key]
        self.current_class_count = self.class_counts[self.current_label]
        self.frame_buffer = []
        self.is_auto_collecting = False
        print(f"\n라벨 선택: {self.current_label} ({GESTURE_NAME[self.current_label]}) - 현재 {self.current_class_count}개")
    
    def _toggle_auto_collect(self):
        """자동 수집 시작/중지"""
        if self.current_label is None:
            print("⚠️ 먼저 라벨을 선택하세요!")
            return
        
        if self.current_class_count >= self.samples_per_class:
            print(f"⚠️ {GESTURE_NAME[self.current_label]}는 이미 {self.samples_per_class}개 수집 완료!")
            return
        
        self.is_auto_collecting = not self.is_auto_collecting
        
        if self.is_auto_collecting:
            self.frame_buffer = []
            self.last_save_time = time.time()
            remaining = self.samples_per_class - self.current_class_count
            print(f"\n🔴 자동 수집 시작: {GESTURE_NAME[self.current_label]} ({remaining}개 남음)")
        else:
            print(f"\n⏸️ 자동 수집 중지")
    
    def _save_sequence(self):
        """시퀀스 저장"""
        if len(self.frame_buffer) >= self.seq_length and self.current_label is not None:
            seq = np.array(self.frame_buffer[-self.seq_length:])  # 최근 30프레임
            self.collected_data.append({
                'sequence': seq.tolist(),
                'label': self.current_label
            })
            
            self.current_class_count += 1
            self.class_counts[self.current_label] = self.current_class_count
            
            remaining = self.samples_per_class - self.current_class_count
            print(f"  💾 {GESTURE_NAME[self.current_label]} #{self.current_class_count} 저장 (남은: {remaining})")
    
    def _save_all(self):
        """전체 데이터 저장"""
        if not self.collected_data:
            print("\n⚠️ 저장할 데이터 없음")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 저장
        json_file = self.output_dir / f"stgcn_data_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.collected_data, f)
        
        # NPY 저장 (학습용)
        sequences = np.array([d['sequence'] for d in self.collected_data])
        labels = np.array([d['label'] for d in self.collected_data])
        
        np.save(self.output_dir / f"sequences_{timestamp}.npy", sequences)
        np.save(self.output_dir / f"labels_{timestamp}.npy", labels)
        
        print(f"\n" + "=" * 50)
        print(f"💾 저장 완료!")
        print(f"  - 총 시퀀스: {len(self.collected_data)}개")
        print(f"  - Shape: {sequences.shape}")
        print(f"  - 파일: {self.output_dir}")
        print("=" * 50)
        
        # 클래스별 통계
        print("\n📊 클래스별 수집 현황:")
        for label, count in sorted(self.class_counts.items()):
            if count > 0:
                status = "✅" if count >= self.samples_per_class else "📌"
                print(f"  {status} {GESTURE_NAME[label]}: {count}개")
    
    def _draw_ui(self, frame):
        """UI 표시"""
        h, w = frame.shape[:2]
        
        # 상단 패널
        cv2.rectangle(frame, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 140), (0, 255, 136), 2)
        
        # 현재 라벨
        if self.current_label is not None:
            label_text = f"Label: {self.current_label} ({GESTURE_NAME[self.current_label]})"
        else:
            label_text = "Label: 선택 안됨"
        cv2.putText(frame, label_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
        
        # 수집 상태
        if self.is_auto_collecting:
            status = "AUTO COLLECTING"
            color = (0, 0, 255)
        else:
            status = "READY (SPACE to start)"
            color = (0, 255, 0)
        cv2.putText(frame, status, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 진행률
        if self.current_label is not None:
            progress = f"Progress: {self.current_class_count}/{self.samples_per_class}"
            cv2.putText(frame, progress, (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 프로그레스 바
            bar_width = int(300 * self.current_class_count / self.samples_per_class)
            cv2.rectangle(frame, (20, 110), (320, 125), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 110), (20 + bar_width, 125), (0, 255, 136), -1)
        
        # 버퍼 상태
        cv2.putText(frame, f"Buffer: {len(self.frame_buffer)}/{self.seq_length}", (20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # 총 수집량
        cv2.putText(frame, f"Total: {len(self.collected_data)}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


if __name__ == "__main__":
    collector = AutoDataCollector(samples_per_class=50)
    collector.run()
