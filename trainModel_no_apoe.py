import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from tensorflow.keras.regularizers import l2

# =========================================================================
# 1. 환경 설정 및 데이터 불러오기/통합
# =========================================================================
print("=" * 70)
print("SCD 제외, APOE 없는 치매 예측 모델 학습")
print("=" * 70)
print("\n--- 1. 데이터 불러오기 및 통합 시작 ---")

# 은서님의 실제 경로
BASE_PATH = "C:/Users/은서/Desktop/Univ/cohort/soul/soul/"

try:
    # 데이터 불러오기
    df_clinical = pd.read_excel(os.path.join(BASE_PATH, "screening_data_1001.xlsx"))
    df_snsb = pd.read_excel(os.path.join(BASE_PATH, "SNSB_1000.xlsx"))

    # ID 컬럼 이름 통일
    if 'Subject ID' in df_snsb.columns:
        df_snsb.rename(columns={'Subject ID': 'SubjectID'}, inplace=True)

    # 데이터 통합 (Inner Join)
    df_final_tabular = pd.merge(df_clinical, df_snsb, on='SubjectID', how='inner')
    df_dnn = df_final_tabular.copy()

    print(f"✅ 데이터 통합 완료. 최종 데이터 크기: {df_dnn.shape[0]} 행\n")

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

# 2.1. 핵심 변수 선택 (APOE 제외)
core_features = [
    'SubjectID', 'dm_10', 'dm_11', 'DIA_01', 'dm_06', 
    'K_MMSE_total_score', 'kiadl_total', 'Digit_span_Forward', 
    'SVLT_recall_total_score', 'RCFT_immediate_recall'
]
df_dnn = df_dnn[core_features].copy()

# 2.2. 목표 변수(Target) 인코딩 - ⚠️ SCD 제외!
# CN: 0, MCI: 1, Dementia: 2
diagnosis_map = {
    'CN': 0, 
    # 'SCD': 0,  # ← SCD 제거!
    'MCI': 1, 
    'Dem': 2,        # ← 실제 데이터의 Dementia 표기
    'Dementia': 2, 
    'AD': 2, 
    'VD': 2, 
    'OTHERS': 2
}
df_dnn['Target'] = df_dnn['DIA_01'].map(diagnosis_map)

print(f"매핑 전 데이터 크기: {df_dnn.shape[0]} 행")
print(f"진단 분포:\n{df_dnn['DIA_01'].value_counts()}")
print(f"\nTarget 매핑 후 결측치 개수: {df_dnn['Target'].isna().sum()}개 (SCD 및 기타)")

df_dnn.drop(columns=['DIA_01'], inplace=True)
df_dnn.dropna(subset=['Target'], inplace=True) 

print(f"\nSCD 제거 후 데이터 크기: {df_dnn.shape[0]} 행")
print(f"Target 분포:\n{df_dnn['Target'].value_counts()}\n")

# 2.3. 성별 인코딩
df_dnn['Sex_Female_1'] = df_dnn['dm_06'].replace({1: 0, 2: 1})
df_dnn.drop(columns=['dm_06'], inplace=True)

# 2.4. 수치형 피쳐 결측치 처리 (평균 대체)
numerical_cols_for_imputation = [
    'dm_10', 'dm_11', 'K_MMSE_total_score', 'kiadl_total', 
    'Digit_span_Forward', 'SVLT_recall_total_score', 'RCFT_immediate_recall'
]
for col in numerical_cols_for_imputation:
    df_dnn[col].fillna(df_dnn[col].mean(), inplace=True)

# =========================================================================
# 3. 표준화 및 인덱스 초기화 (KeyError 해결)
# =========================================================================
print("--- 3. 표준화 및 인덱스 초기화 시작 ---")

numerical_cols = numerical_cols_for_imputation 
scaler = StandardScaler()
df_dnn[numerical_cols] = scaler.fit_transform(df_dnn[numerical_cols])

# 💥 FIX: KeyError 방지를 위한 인덱스 초기화
df_dnn.reset_index(drop=True, inplace=True) 
print("✅ 인덱스 초기화 완료.\n")

# =========================================================================
# 4. 층화추출 기반 데이터 분할 (Split)
# =========================================================================
print("--- 4. 층화추출 기반 데이터 분할 시작 ---")

X = df_dnn.drop(columns=['SubjectID', 'Target']).copy()
y = df_dnn['Target']

# Train/Validation (80%) 와 Test (20%) 분할
sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_val_index, test_index in sss_test.split(X, y):
    df_test = df_dnn.loc[test_index].copy()
    df_test['Data_Set'] = 'Test'
    X_train_val = X.iloc[train_val_index]
    y_train_val = y.iloc[train_val_index]

# Train (64%) 와 Validation (16%) 분할
sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, val_index in sss_val.split(X_train_val, y_train_val):
    df_train = df_dnn.loc[train_index].copy()
    df_validation = df_dnn.loc[val_index].copy()
    
    df_train['Data_Set'] = 'Train'
    df_validation['Data_Set'] = 'Validation'
    
    # 모델 학습용 X, y 정의
    X_train = df_train.drop(columns=['SubjectID', 'Target', 'Data_Set'])
    y_train = df_train['Target']
    X_val = df_validation.drop(columns=['SubjectID', 'Target', 'Data_Set'])
    y_val = df_validation['Target']
    X_test = df_test.drop(columns=['SubjectID', 'Target', 'Data_Set'])
    y_test = df_test['Target']

print(f"훈련 세트: {df_train.shape[0]}, 검증 세트: {df_validation.shape[0]}, 테스트 세트: {df_test.shape[0]}")
print(f"\n입력 피처 개수: {X_train.shape[1]}")
print(f"피처 목록: {list(X_train.columns)}\n")

# =========================================================================
# 5. 최종 데이터셋 저장
# =========================================================================
print("--- 5. 최종 데이터셋 저장 시작 ---")

df_train.to_csv('dnn_tabular_train_data_no_scd.csv', index=False, encoding='utf-8')
df_validation.to_csv('dnn_tabular_validation_data_no_scd.csv', index=False, encoding='utf-8')
df_test.to_csv('dnn_tabular_test_data_no_scd.csv', index=False, encoding='utf-8')
df_master = pd.concat([df_train, df_validation, df_test], axis=0)
df_master.to_csv('dnn_tabular_master_data_no_scd.csv', index=False, encoding='utf-8')

print("✅ CSV 파일 4개 저장 완료.\n")

# =========================================================================
# 6. DNN 모델 정의 및 학습 준비
# =========================================================================
print("--- 6. DNN 모델 정의 및 학습 준비 ---")

# 6.1. 클래스 및 피처 개수 정의
INPUT_FEATURES = X_train.shape[1] 
NUM_CLASSES = len(y_train.unique()) 

print(f"입력 피처 수: {INPUT_FEATURES}")
print(f"클래스 수: {NUM_CLASSES}")
print(f"클래스 레이블: {sorted(y_train.unique())}\n")

# 6.2. 클래스 가중치 계산 및 샘플 가중치 배열 생성
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train.to_numpy()
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
sample_weights = np.array([class_weight_dict[label] for label in y_train])

print(f"클래스 가중치: {class_weight_dict}\n")

# 6.3. DNN 모델 아키텍처 정의
def build_tabular_dnn(input_shape, feature_vector_dim=64, num_classes=3):
    input_layer = Input(shape=(input_shape,), name='tabular_input')
    
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 특징 추출층
    feature_vector = Dense(feature_vector_dim, activation='relu', name='tabular_feature_vector')(x)
    
    # 최종 분류 출력층
    classifier_output = Dense(num_classes, activation='softmax', name='tabular_classification_output')(feature_vector)
    
    model = Model(inputs=input_layer, outputs=[feature_vector, classifier_output])
    return model

# 모델 생성 및 컴파일
dnn_model = build_tabular_dnn(input_shape=INPUT_FEATURES, num_classes=NUM_CLASSES)

dnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss={
        'tabular_feature_vector': None,  
        'tabular_classification_output': 'sparse_categorical_crossentropy'
    },
    metrics={'tabular_classification_output': 'accuracy'}
)

print("--- DNN 모델 컴파일 완료 ---")
print(dnn_model.summary())

# =========================================================================
# 7. 모델 학습 (Training)
# =========================================================================
print("\n" + "=" * 70)
print("--- 7. 모델 학습 시작 ---")
print("=" * 70 + "\n")

early_stopping = EarlyStopping(
    monitor='val_tabular_classification_output_loss', 
    patience=20, 
    restore_best_weights=True,
    mode='min',
    verbose=1
)

checkpoint_filepath = 'best_dnn_tabular_model_no_scd.h5'
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_tabular_classification_output_loss',
    save_best_only=True,
    verbose=1,
    mode='min'
)

# 모델 학습
history = dnn_model.fit(
    X_train, 
    (None, y_train),
    
    validation_data=(
        X_val, 
        (None, y_val) 
    ),
    
    epochs=150, 
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint],
    
    sample_weight=(None, sample_weights), 
    
    verbose=1
)

print(f"\n✅ 모델 학습 완료. 최적 모델은 '{checkpoint_filepath}'에 저장되었습니다.")

# =========================================================================
# 8. 테스트 데이터셋 성능 평가
# =========================================================================
print("\n" + "=" * 70)
print("--- 8. 테스트 데이터셋 성능 평가 ---")
print("=" * 70 + "\n")

try:
    # 최적 모델 로드 및 컴파일
    best_model = tf.keras.models.load_model(checkpoint_filepath, compile=False)
    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss={
            'tabular_feature_vector': None,  
            'tabular_classification_output': 'sparse_categorical_crossentropy'
        },
        metrics={'tabular_classification_output': 'accuracy'}
    )
    
    # evaluate() 실행
    evaluation_results = best_model.evaluate(
        X_test, 
        (None, y_test),
        verbose=0
    )
    
    # 결과 출력
    metrics_names = best_model.metrics_names
    
    if len(evaluation_results) >= 2:
        total_loss = evaluation_results[0]
        accuracy = evaluation_results[-1] 
        
        print(f"📊 테스트 데이터 평가 결과:")
        print(f"   Total Loss: {total_loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
    else:
        print(f"경고: 평가 지표(Accuracy)를 찾을 수 없습니다.")
        print(f"Total Loss: {evaluation_results[0]:.4f}")
        print(f"Keras Metrics Names: {metrics_names}")
    
    # 추가 분석: 클래스별 예측 결과
    print("\n--- 클래스별 예측 분석 ---")
    _, y_pred_probs = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n혼동 행렬 (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n분류 리포트 (Classification Report):")
    target_names = ['CN (0)', 'MCI (1)', 'Dementia (2)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
except Exception as e:
    print(f"테스트 데이터 평가 중 오류 발생: {e}")

# =========================================================================
# 9. Scaler 저장
# =========================================================================
import pickle

scaler_filepath = 'dnn_scaler_object_no_scd.pkl'
try:
    with open(scaler_filepath, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"\n✅ StandardScaler 객체가 '{scaler_filepath}' 파일로 저장되었습니다.")
except Exception as e:
    print(f"\n❌ StandardScaler 객체 저장 중 오류 발생: {e}")

print("\n" + "=" * 70)
print("🎉 모든 작업 완료!")
print("=" * 70)
print(f"\n생성된 파일:")
print(f"  - {checkpoint_filepath}")
print(f"  - {scaler_filepath}")
print(f"  - dnn_tabular_*_no_scd.csv (4개 파일)")