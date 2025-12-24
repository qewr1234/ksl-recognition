from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

def download_whisper_model(model_size="medium"):
    """
    Whisper 모델 다운로드
    model_size: "tiny", "base", "small", "medium", "large-v3"
    - tiny: 가장 빠름, 정확도 낮음 (~39M)
    - small: 균형 잡힘 (~244M) [추천]
    - medium: 더 정확 (~769M)
    - large-v3: 가장 정확, 느림 (~1.5G)
    """
    
    model_name = f"openai/whisper-{model_size}"
    save_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(save_dir, f"whisper-{model_size}")
    
    print(f"모델 다운로드 중: {model_name}")
    print(f"저장 경로: {model_path}")
    print("-" * 50)
    
    # Processor 다운로드 (토크나이저 + 피처 추출기)
    print("1/2 Processor 다운로드 중...")
    processor = WhisperProcessor.from_pretrained(model_name)
    processor.save_pretrained(model_path)
    print("✓ Processor 완료")
    
    # Model 다운로드
    print("2/2 Model 다운로드 중...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.save_pretrained(model_path)
    print("✓ Model 완료")
    
    print("-" * 50)
    print(f"다운로드 완료! 저장 위치: {model_path}")
    
    return model_path

if __name__ == "__main__":
    # small 모델 추천 (속도와 정확도 균형)
    # 더 빠른 걸 원하면 "tiny" 또는 "base"
    # 더 정확한 걸 원하면 "medium" 또는 "large-v3"
    download_whisper_model("medium")
