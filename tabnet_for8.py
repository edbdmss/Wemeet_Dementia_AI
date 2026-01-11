import pandas as pd
import numpy as np
import os
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# 1. 데이터 로드 (경로는 본인의 환경에 맞게 수정하세요)
BASE_PATH = "models/"

try:
    df_clinical = pd.read_excel(os.path.join(BASE_PATH, "screening_data_1001.xlsx"))
    df_snsb = pd.read_excel(os.path.join(BASE_PATH, "SNSB_1000.xlsx"))
    df_snsb.rename(columns={'Subject ID': 'SubjectID'}, inplace=True)
    df_final = pd.merge(df_clinical, df_snsb, on='SubjectID', how='inner')
except Exception as e:
    print(f"❌ 데이터 로드 실패: {e}")
    exit()

# 2. 전처리 (DNN과 동일한 8개 변수 규격 맞추기)
feature_cols = [
    'dm_10', 'dm_11', 'K_MMSE_total_score', 'kiadl_total',
    'Digit_span_Forward', 'SVLT_recall_total_score', 'RCFT_immediate_recall'
]
df_final['Sex_Female_1'] = df_final['dm_06'].replace({1: 0, 2: 1})
all_features = feature_cols + ['Sex_Female_1']

# 타겟 매핑 (SCD 제외)
diagnosis_map = {'CN': 0, 'MCI': 1, 'Dem': 2, 'Dementia': 2, 'AD': 2}
df_final['Target'] = df_final['DIA_01'].map(diagnosis_map)
df_final.dropna(subset=['Target'], inplace=True)

X = df_final[all_features].copy()
y = df_final['Target'].values.astype(int)

# 💥 [중요] 결측치 평균 처리 및 스케일링 (기존에 저장된 스케일러가 있다면 그것을 써도 좋지만, 여기서는 새로 맞춥니다)
X = X.fillna(X.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. TabNet 5개 Seed 학습 (515개용 모델을 8개용으로 덮어쓰기)
seeds = [42, 2023, 2024, 777, 999]
if not os.path.exists('models'): os.makedirs('models')

for s in seeds:
    print(f"🚀 TabNet Seed {s} 학습 시작 (8개 변수용)...")
    clf = TabNetClassifier(seed=s, device_name='cpu', verbose=0)
    clf.fit(
        X_train=X_scaled, y_train=y,
        max_epochs=50, batch_size=64
    )
    # 기존 models 폴더에 저장 (기존 515개짜리 파일이 이 파일로 대체됩니다)
    clf.save_model(f"models/tabnet_model_seed_{s}")

print("\n✅ TabNet 모델 5개 모두 8개 변수용으로 업데이트 완료!")
print("이제 바로 'integration.py'를 실행하시면 됩니다.")