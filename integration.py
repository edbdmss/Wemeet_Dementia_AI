import pandas as pd
import numpy as np
import os
import tensorflow as tf
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 1. 경로 설정
MODEL_DIR = "models"
DNN_MODEL_PATH = "best_dnn_tabular_model_no_scd.h5"
TEST_DATA_PATH = "dnn_tabular_test_data_no_scd.csv"

# 피처 순서 고정
numerical_cols = [
    'dm_10', 'dm_11', 'K_MMSE_total_score', 'kiadl_total',
    'Digit_span_Forward', 'SVLT_recall_total_score', 'RCFT_immediate_recall'
]
categorical_cols = ['Sex_Female_1']
feature_cols = numerical_cols + categorical_cols

# 2. 데이터 로드 (이미 스케일링 된 상태)
try:
    df_test = pd.read_csv(TEST_DATA_PATH)
    X_test_final = df_test[feature_cols].values
    y_test = df_test['Target'].values
    print(f" 데이터 로드 완료. (Shape: {X_test_final.shape})")
except Exception as e:
    print(f" 데이터 로드 에러: {e}")
    exit()

# 3. 모델 로드
print("---모델 로딩 중---")
dnn_model = tf.keras.models.load_model(DNN_MODEL_PATH, compile=False)

tabnet_models = []
for seed in [42, 2023, 2024, 777, 999]:
    clf = TabNetClassifier()
    clf.load_model(os.path.join(MODEL_DIR, f"tabnet_model_seed_{seed}.zip"))
    tabnet_models.append(clf)

# 4. 예측 
print("--- 분석 시작 ---")

# --- DNN 예측 부분 수정 ---
dnn_out = dnn_model.predict(X_test_final, verbose=0)

# 결과가 리스트일 경우 확률값(보통 마지막 또는 첫번째)을 배열로 변환
if isinstance(dnn_out, list):
    dnn_probs = np.array(dnn_out[-1])
else:
    dnn_probs = dnn_out

# 차원 맞추기 (2차원 배열임)
if len(dnn_probs.shape) == 3:
    dnn_probs = np.squeeze(dnn_probs, axis=1)
elif len(dnn_probs.shape) == 1:
    pass

# TabNet 앙상블 예측
tabnet_probs = np.mean([clf.predict_proba(X_test_final) for clf in tabnet_models], axis=0)

# 최종 앙상블 (5:5)
final_probs = (dnn_probs * 0.5) + (tabnet_probs * 0.5)
final_preds = np.argmax(final_probs, axis=1)

# 5. 결과 출력
print("\n" + "="*60)
print(f" 최종 통합 앙상블 정확도: {accuracy_score(y_test, final_preds)*100:.2f}%")
print("="*60)
print(classification_report(y_test, final_preds, target_names=['Normal (CN)', 'MCI', 'Dementia']))

print("\n📉 혼동 행렬:")
print(confusion_matrix(y_test, final_preds))
