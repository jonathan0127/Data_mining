# -*- coding: utf-8 -*-
"""
Created on Fri May 30 17:26:29 2025

@author: HP
"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from params import datatype

def predict_with_oneclass_svm(model, svm_model, scaler, test_data, label_encoder):

    test_data_scaled = scaler.transform(test_data)
    # 使用 One-Class SVM 預測異常（-1 為異常，1 為正常）
    anomaly_scores = svm_model.decision_function(test_data_scaled)
    is_anomaly = svm_model.predict(test_data_scaled) == -1
    
    predictions = []
    for i, is_unknown in enumerate(is_anomaly):
        if is_unknown:
            predictions.append('unknown')
        else:
            single_sample = test_data.iloc[[i]]
            pred = model.predict(single_sample)[0]
            class_name = label_encoder.inverse_transform([pred])[0]
            predictions.append(class_name)
    
    return pd.Series(predictions), anomaly_scores

def write_col_name_A():
    col_name=['Age','Sex','Height','Weight','QRS_duration',
          'P-R_interval','Q-T_interval','T_interval','P_interval',
          'QRS_vector_angle','T_vector_angle','P_vector_angle','QRST_vector_angle',
          'J_vector_angle','Heart_rate']
    channels=['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
    for ch in channels:
        col_name+=[f"{ch}_Q_width",f"{ch}_R_width",f"{ch}_S_width",f"{ch}_R'_width",
                   f"{ch}_S'_width",f"{ch}_deflection_number",f"{ch}_Ragged_R",
                   f"{ch}_R_diphasic_derivation",f"{ch}_Ragged_P",f"{ch}_P_diphasic_derivation",
                   f"{ch}_Ragged_T",f"{ch}_T_diphasic_derivation"]
    for ch in channels:
        col_name+=[f"{ch}_JJ_Amp",f"{ch}_Q_Amp",f"{ch}_R_Amp",f"{ch}_S_Amp",
                   f"{ch}_R'_Amp",f"{ch}_S'_Amp",f"{ch}_P_Amp",f"{ch}_T_Amp",
                   f"{ch}_QRSA_Amp",f"{ch}_QRSTA_Amp"]
    return col_name

def select_core_features_A(data: pd.DataFrame):
    core_features = [
        'Age', 'Sex', 'Height', 'Weight',
        'QRS_duration', 'Q-T_interval', 'Heart_rate', 'P-R_interval',
        'QRS_vector_angle', 'T_vector_angle', 'P_vector_angle', 'QRST_vector_angle', 'J_vector_angle',
        'DI_Q_width', 'DI_R_width', 'DI_S_width', 'DI_deflection_number',
        'DI_Ragged_R', 'DI_R_diphasic_derivation', 'DI_Q_Amp', 'DI_R_Amp',
        'DI_S_Amp', 'DI_P_Amp', 'DI_T_Amp', 'DI_QRSA_Amp',
        'V1_R_Amp', 'V1_S_Amp', 'V1_P_Amp', 'V1_T_Amp'
    ]
    
    available_features = [col for col in core_features if col in data.columns]
    return data[available_features]

#Drop掉缺失值超過75%的欄位
def drop_col(data: pd.DataFrame, threshold):
    missing_ratio=data.isnull().mean()
    dropping_cols=missing_ratio[missing_ratio>threshold].index
    return data.drop(columns=dropping_cols)

#補每欄的缺失值
def fill_missing(data: pd.DataFrame):
    filling_value={}
    for col in data:
        if data[col].dtype in ['float64','int64']:
            filling_value[col]=data[col].median()
        else:
            filling_value[col]=data[col].mode()[0]
    return filling_value


#找出已知類(如果該訓練標籤中此類的樣本數少於5就分到未知類，其他為已知類)
def merge_rare_classes_to_unknown(labels: pd.Series):
    class_counts = labels.value_counts()
    min_samples=5
    rare_classes = class_counts[class_counts < min_samples].index
    labels_merged = labels.apply(lambda x: 'unknown' if x in rare_classes else x)
    return labels_merged


#標準化: 若分類演算法需要可依需求取用
def Standardlize(data: pd.DataFrame):
    scaler=StandardScaler()
    data_scaled=scaler.fit_transform(data)
    return data_scaled, scaler

#標準化線性特徵
def standardize_numeric_features(train_data, test_data):
    numeric_features = train_data.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    # 複製數據避免修改原始數據
    train_data_scaled = train_data.copy()
    test_data_scaled = test_data.copy()
    
    # 對數值型特徵進行標準化
    train_data_scaled[numeric_features] = scaler.fit_transform(train_data[numeric_features])
    test_data_scaled[numeric_features] = scaler.transform(test_data[numeric_features])
    
    print(f"標準化了 {len(numeric_features)} 個數值型特徵")
    return train_data_scaled, test_data_scaled, scaler

#對訓練集做Smote來讓每個類別的樣本數比較平衡
def Smote(data1: pd.DataFrame, data2: pd.DataFrame):
    smote=SMOTE(random_state=42)
    data1, data2=smote.fit_resample(data1,data2)
    return data1, data2

#使用PCA進行降維
def apply_pca(train_data, test_data, n_components=0.95):
    pca = PCA(n_components=n_components, random_state=42)
    
    # 只對訓練數據進行PCA
    train_data_pca = pca.fit_transform(train_data)
    
    test_data_pca = pca.transform(test_data)
    
    pca_columns = [f'PC{i+1}' for i in range(train_data_pca.shape[1])]
    train_data_pca = pd.DataFrame(train_data_pca, columns=pca_columns)
    test_data_pca = pd.DataFrame(test_data_pca, columns=pca_columns)
    
    print(f"PCA降維：從 {train_data.shape[1]} 維降至 {train_data_pca.shape[1]} 維")
    
    return train_data_pca, test_data_pca, pca

def subprocess_dataA():
    col_name=write_col_name_A()

    train_data=pd.read_csv('dataSet/dataA/Arrhythmia_train_data.csv',header=None,names=col_name)
    train_label=pd.read_csv('dataSet/dataA/Arrhythmia_train_label.csv',header=None,names=['Class'])
    test_data=pd.read_csv('dataSet/dataA/Arrhythmia_test_data.csv',header=None,names=col_name)
    test_label=pd.read_csv('dataSet/dataA/Arrhythmia_test_label.csv',header=None,names=['Class'])
    train_data=drop_col(train_data,0.75)
    test_data=test_data[train_data.columns]     

    # train_data = select_core_features_A(train_data)
    # test_data = select_core_features_A(test_data)
    
    return{
        "train_data": train_data,
        "train_label": train_label,
        "test_data": test_data,
        "test_label": test_label
    }

def subprocess_dataB():

    train_data=pd.read_csv('dataSet/dataB/train_data.csv',header=0)
    train_label=pd.read_csv('dataSet/dataB/train_label.csv',header=0)
    test_data=pd.read_csv('dataSet/dataB/test_data.csv',header=0)
    test_label=pd.read_csv('dataSet/dataB/test_label.csv',header=0)

    train_data = train_data.iloc[:, 1:]
    train_label = train_label.iloc[:, 1:]
    test_data = test_data.iloc[:, 1:]
    test_label = test_label.iloc[:, 1:]

    train_data=drop_col(train_data,0.75)
    test_data=test_data[train_data.columns]     

    print(train_data.head())
    print(train_label.head())

    return{
        "train_data": train_data,
        "train_label": train_label,
        "test_data": test_data,
        "test_label": test_label
    }

#資料讀取+整合過後的資料預處理
def data_read_and_preprocess(pca, standard):

    if datatype == 'A':
        retData = subprocess_dataA()
    else:
        retData = subprocess_dataB()
    
    train_data = retData['train_data']
    train_label = retData['train_label']
    test_data = retData['test_data']
    test_label = retData['test_label']
    
    
    filled_values=fill_missing(train_data)
    train_data=train_data.fillna(filled_values)
    test_data=test_data.fillna(filled_values)   
    
    if standard:
        train_data, test_data, feature_scaler = standardize_numeric_features(train_data, test_data)
    
    
    
    train_label['Class'] = merge_rare_classes_to_unknown(train_label['Class'])
    
    class_all=train_label['Class'].unique()
    test_label_processed=test_label['Class'].apply(lambda x: x if x in class_all else 'unknown')
    test_label_fulled=test_label['Class']
    is_known = train_label['Class'] != 'unknown'
    known_data = train_data[is_known].reset_index(drop=True)
    known_label = train_label[is_known].reset_index(drop=True)
    
    #Data preprocessing: Tranfer the class label to from str to int
    le=LabelEncoder()
    known_label_encoded=le.fit_transform(known_label['Class'])
    known_data_smote, known_label_encoded_smote=Smote(known_data,known_label_encoded)
    
    
    if pca:
        known_data_pca, test_data_pca, pca_model = apply_pca(known_data_smote, test_data, n_components=0.95)
        known_data_smote = known_data_pca
        test_data = test_data_pca
    else:
        pca_model = None
    
    return{
        "train_data": known_data_smote,
        "train_label_encoded": known_label_encoded_smote,
        "test_data": test_data,
        "test_label_processed": test_label_processed,
        "label_encoder": le,
        "test_label_fulled" : test_label_fulled,
        "pca_model": pca_model
        }







