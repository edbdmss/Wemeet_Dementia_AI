# [팀원 공유용] TabNet 앙상블 모델 실행 코드
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np

def predict_dementia_stage(X_input):
    """
    X_input: 전처리가 완료된 2차원 Numpy 배열 (Shape: [샘플수, Feature수])
    반환값: 각 클래스별 확률 [CN확률, SCD확률, MCI확률, Dem확률]
    """
    # 1. 모델 파일이 있는 경로 (팀원 컴퓨터 경로에 맞게 수정 필요)
    model_dir = './models/' 
    seeds = [42, 2023, 2024, 777, 999]
    preds_probs = []

    print("🧠 TabNet 앙상블 모델이 예측 중입니다...")

    # 2. 5개 모델을 순서대로 불러와서 예측
    for seed in seeds:
        clf = TabNetClassifier()
        # 파일명 주의: 팀원이 저장한 파일명과 같아야 함
        clf.load_model(f"{model_dir}tabnet_model_seed_{seed}.zip")
        
        # 확률 예측
        pred = clf.predict_proba(X_input)
        preds_probs.append(pred)

    # 3. 5개 결과 평균 (앙상블)
    avg_prob = np.mean(preds_probs, axis=0)
    
    return avg_prob

# 사용 예시
# final_prob = predict_dementia_stage(X_test_data)
# print(final_prob)