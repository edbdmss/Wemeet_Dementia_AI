# TabNet 앙상블 모델 실행 코드
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import os 

def predict_dementia_stage(X_input):
    """
    X_input: 전처리가 완료된 2차원 Numpy 배열 (Shape: [샘플수, Feature수])
    반환값: 각 클래스별 확률 [CN확률, SCD확률, MCI확률, Dem확률]
    """
    
    # 1. 모델 파일이 있는 폴더 경로
    model_dir = '/Users/yueun/Desktop/Wemeet/tabnet_result_model' 
    seeds = [42, 2023, 2024, 777, 999]
    preds_probs = []

    print("TabNet 앙상블 모델이 예측 중")

    # 2. 5개 모델을 순서대로 불러와서 예측
    for seed in seeds:
        clf = TabNetClassifier()
        filename = f"tabnet_model_seed_{seed}.zip"
        full_path = os.path.join(model_dir, filename)
        
        # 불러오기 (혹시 에러나면 경로 출력해서 확인)
        try:
            clf.load_model(full_path)
        except Exception as e:
            print(f"모델 로드 실패: {full_path}")
            print(f"에러 메시지: {e}")
            raise e
        
        # 확률 예측
        pred = clf.predict_proba(X_input)
        preds_probs.append(pred)

    # 3. 5개 결과 평균 (앙상블 - Soft Voting)
    avg_prob = np.mean(preds_probs, axis=0)
    
    return avg_prob


# 사용 예시
# 데이터 파일이 있는 경로 (수정 필요 시 변경)
data_dir = '//Users/yueun/Desktop/Wemeet/test_data'
data_path = os.path.join(data_dir, 'X_test_data.npy')

# 데이터 파일이 존재할 경우에만 실행
if os.path.exists(data_path):
    X_test_data = np.load(data_path)
    final_prob = predict_dementia_stage(X_test_data)
    print(final_prob)
else:
    print(f"데이터 파일이 없습니다: {data_path}")