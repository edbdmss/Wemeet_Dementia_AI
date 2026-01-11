import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
# TensorFlow는 모델 정의 및 클래스 가중치 계산 확인용으로 포함
import tensorflow as tf 
import os

# 파일 경로 (사용자님께서 제공하신 절대 경로 사용)
BASE_PATH = "C:/Users/rlatj/OneDrive/Desktop/cohort/soul/soul/"

# =========================================================================
# 1. 데이터 불러오기 및 통합 (사용자 코드)
# =========================================================================
print("--- 1. 데이터 불러오기 및 통합 시작 ---")

try:
    # pd.read_excel을 사용하여 .xlsx 파일 로드 (첫 번째 시트 가정)
    df_clinical = pd.read_excel(os.path.join(BASE_PATH, "screening_data_1001.xlsx"))
    df_snsb = pd.read_excel(os.path.join(BASE_PATH, "SNSB_1000.xlsx"))
    df_apoe = pd.read_excel(os.path.join(BASE_PATH, "APOE_982.xlsx"))

    print(f"임상/인구통계 데이터 (df_clinical): {df_clinical.shape[0]} 행")
    print(f"SNSB 데이터 (df_snsb): {df_snsb.shape[0]} 행")
    print(f"APOE 데이터 (df_apoe): {df_apoe.shape[0]} 행")

    # 통합을 위한 핵심 ID 컬럼 이름 통일
    if 'Subject ID' in df_snsb.columns:
        df_snsb.rename(columns={'Subject ID': 'SubjectID'}, inplace=True)
    if 'Subject_ID' in df_apoe.columns:
        df_apoe.rename(columns={'Subject_ID': 'SubjectID'}, inplace=True)

    # 1차 병합: 임상/인구통계 + APOE (SubjectID 기준)
    df_merged_1 = pd.merge(df_clinical, df_apoe[['SubjectID', 'APOE']], on='SubjectID', how='inner')

    # 2차 병합: 1차 병합 결과 + SNSB (SubjectID 기준)
    df_final_tabular = pd.merge(df_merged_1, df_snsb, on='SubjectID', how='inner')
    df_dnn = df_final_tabular.copy()

    print(f"✅ 데이터 통합 완료. 최종 통합 데이터셋 크기: {df_dnn.shape[0]} 행\n")

except FileNotFoundError:
    print(f"오류: 지정된 경로({BASE_PATH})에서 파일을 찾을 수 없습니다. 경로를 확인하십시오.")
    exit()
except Exception as e:
    print(f"데이터 로드 또는 통합 중 오류 발생: {e}")
    exit()


# =========================================================================
# 2. 전처리: 변수 선택, 인코딩, 결측치 처리
# =========================================================================
print("--- 2. 전처리 (변수 인코딩 및 결측치 처리) 시작 ---")

# 2.1. 핵심 변수 선택 (DNN 입력 피처)
# 실제 구현 시에는 모든 필요한 SNSB 원점수 컬럼을 포함해야 합니다.
core_features = [
    'SubjectID', 'dm_10', 'dm_11', 'APOE', 'DIA_01', 'dm_06', # 핵심 인구통계/진단/유전자
    'K_MMSE_total_score', 'kiadl_total', 'Digit_span_Forward', # 핵심 임상/인지 점수
    'SVLT_recall_total_score', 'RCFT_immediate_recall' # 대표 SNSB 점수
    # 필요한 SNSB 컬럼을 더 추가하세요. (예: Naming_K_BNT, Praxi_Ideomotor 등)
]
df_dnn = df_dnn[core_features].copy()


# 2.2. 목표 변수(Target) 인코딩 (CN/SCD: 0, MCI: 1, Dementia: 2)
'''
diagnosis_map = {
    'CN': 0, 'SCD': 0, 'MCI': 1, 'Dementia': 2, 
    'AD': 2, 'VD': 2, 'OTHERS': 2 
}
df_dnn['Target'] = df_dnn['DIA_01'].map(diagnosis_map)
df_dnn.drop(columns=['DIA_01'], inplace=True)
df_dnn.dropna(subset=['Target'], inplace=True) # Target 결측치는 제거 '''

# SCD 제외 버전
diagnosis_map = {
    'CN': 0, 'MCI': 1, 'Dementia': 2, 
    'AD': 2, 'VD': 2, 'OTHERS': 2 
}
df_dnn['Target'] = df_dnn['DIA_01'].map(diagnosis_map)
df_dnn.drop(columns=['DIA_01'], inplace=True)
df_dnn.dropna(subset=['Target'], inplace=True)  # SCD는 매핑되지 않아 NaN이 되므로 자동 제거됨

# 2.3. APOE e4 유전자형 인코딩 (e4 보유: 1, 비보유: 0)
df_dnn['APOE_e4_status'] = np.where(df_dnn['APOE'].astype(str).str.contains('4', na=False), 1, 0)
df_dnn.drop(columns=['APOE'], inplace=True)

# 2.4. 성별(dm_06) 이진 인코딩 (1: 남성 -> 0, 2: 여성 -> 1)
df_dnn['Sex_Female_1'] = df_dnn['dm_06'].replace({1: 0, 2: 1})
df_dnn.drop(columns=['dm_06'], inplace=True)


# 2.5. 수치형 피쳐 결측치 처리 (평균 대체)
numerical_cols_for_imputation = [
    'dm_10', 'dm_11', 'K_MMSE_total_score', 'kiadl_total', 
    'Digit_span_Forward', 'SVLT_recall_total_score', 'RCFT_immediate_recall'
]
for col in numerical_cols_for_imputation:
    df_dnn[col].fillna(df_dnn[col].mean(), inplace=True)

print(f"전처리 및 Target 결측치 제거 후 데이터 크기: {df_dnn.shape[0]} 행")

# =========================================================================
# 3. 수치형 데이터 표준화 (Z-Score Standardization)
# =========================================================================
print("--- 3. 수치형 데이터 표준화 시작 ---")

# 표준화 대상 컬럼 정의
numerical_cols = [
    'dm_10', 'dm_11', 'K_MMSE_total_score', 'kiadl_total', 
    'Digit_span_Forward', 'SVLT_recall_total_score', 'RCFT_immediate_recall'
]

scaler = StandardScaler()
df_dnn[numerical_cols] = scaler.fit_transform(df_dnn[numerical_cols])

print(f"✅ 표준화 완료.\n")

# =========================================================================
# ⚠️ 오류 해결: 인덱스 초기화
# dropna 등으로 인해 불연속적인 인덱스를 순차적인 인덱스로 재설정합니다.
# 이로써 StratifiedShuffleSplit이 반환하는 인덱스를 df_dnn이 인식하게 됩니다.
# =========================================================================
df_dnn.reset_index(drop=True, inplace=True)
print("--- 인덱스 초기화 완료. (KeyError 해결 조치) ---\n")


# =========================================================================
# 4. 층화추출 기반 데이터 분할 (Train/Validation/Test)
# =========================================================================
print("--- 4. 층화추출 기반 데이터 분할 시작 (Train: 64%, Validation: 16%, Test: 20%) ---")

X = df_dnn.drop(columns=['SubjectID', 'Target']).copy()
y = df_dnn['Target']

# 4.1. Train/Validation (80%) 와 Test (20%) 분할
sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_val_index, test_index in sss_test.split(X, y):
    # 인덱스가 초기화되었기 때문에 .loc[test_index] 사용 가능
    df_test = df_dnn.loc[test_index].copy() 
    df_test['Data_Set'] = 'Test'
    
    # Train/Val 세트 준비
    X_train_val = X.iloc[train_val_index]
    y_train_val = y.iloc[train_val_index]

# 4.2. Train (64%) 와 Validation (16%) 분할
sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, val_index in sss_val.split(X_train_val, y_train_val):
    # 인덱스가 초기화되었기 때문에 .loc[train_index]/.loc[val_index] 사용 가능
    df_train = df_dnn.loc[train_index].copy()
    df_validation = df_dnn.loc[val_index].copy()
    
    df_train['Data_Set'] = 'Train'
    df_validation['Data_Set'] = 'Validation'
    
    # 모델 학습에 사용할 최종 X, y 정의
    X_train = df_train.drop(columns=['SubjectID', 'Target', 'Data_Set'])
    y_train = df_train['Target']
    X_val = df_validation.drop(columns=['SubjectID', 'Target', 'Data_Set'])
    y_val = df_validation['Target']
    X_test = df_test.drop(columns=['SubjectID', 'Target', 'Data_Set'])
    y_test = df_test['Target']

print(f"훈련 세트 크기 (Train): {df_train.shape[0]} 행")
print(f"검증 세트 크기 (Validation): {df_validation.shape[0]} 행")
print(f"테스트 세트 크기 (Test): {df_test.shape[0]} 행")


# =========================================================================
# 5. 최종 데이터셋 저장 (개별 파일 및 마스터 파일)
# =========================================================================
print("\n--- 5. 최종 데이터셋 저장 시작 ---")

# 5.1. 개별 파일 저장
df_train.to_csv('dnn_tabular_train_data.csv', index=False, encoding='utf-8')
df_validation.to_csv('dnn_tabular_validation_data.csv', index=False, encoding='utf-8')
df_test.to_csv('dnn_tabular_test_data.csv', index=False, encoding='utf-8')

# 5.2. 마스터 파일로 통합 저장
df_master = pd.concat([df_train, df_validation, df_test], axis=0)
df_master.to_csv('dnn_tabular_master_data.csv', index=False, encoding='utf-8')

print("\n✅ 데이터 전처리 파이프라인 완료! CSV 파일이 성공적으로 저장되었습니다.")

import pickle
import os

# ⚠️ 'scaler' 객체가 위에 있는 데이터 전처리 단계에서 정의되었다고 가정합니다.

# 1. 저장할 파일 경로 설정 (h5 파일이 있는 같은 위치 권장)
scaler_filepath = 'dnn_scaler_object.pkl'

# 2. scaler 객체를 파일로 저장
try:
    with open(scaler_filepath, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"\n✅ StandardScaler 객체가 '{scaler_filepath}' 파일로 성공적으로 저장되었습니다.")
except Exception as e:
    print(f"\n❌ StandardScaler 객체 저장 중 오류 발생: {e}")

# 3. 통합 담당자가 로드할 때의 예시 코드 (참고용)
# with open('dnn_scaler_object.pkl', 'rb') as file:
#     loaded_scaler = pickle.load(file)
# X_test_scaled = loaded_scaler.transform(X_test_raw)

