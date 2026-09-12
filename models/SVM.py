import pandas as pd
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def train_svm(train_data, train_label):
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    
    model = SVC(random_state=42, probability=True)
    param_grid = {'C': [0.1, 1, 10, 100],'kernel': ['linear', 'rbf'],'gamma': ['scale', 0.01, 0.1],'class_weight': [None, 'balanced']}
    
    grid_search = GridSearchCV(model, param_grid, scoring='f1_macro', cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(train_data_scaled, train_label)

    print("Best parameter combination:", grid_search.best_params_)
    return grid_search.best_estimator_, scaler

def train_oneclass_svm(train_data):
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    oneclass_svm = OneClassSVM(kernel='rbf',gamma='scale',nu=0.15)
    oneclass_svm.fit(train_data_scaled)
    return oneclass_svm, scaler

def predict_with_oneclass_svm(svm_model, svm_scaler, oneclass_model, oneclass_scaler, test_data, label_encoder):

    test_data_scaled_oneclass = oneclass_scaler.transform(test_data)
    test_data_scaled_svm = svm_scaler.transform(test_data)
    
    anomaly_scores = oneclass_model.decision_function(test_data_scaled_oneclass)
    is_anomaly = oneclass_model.predict(test_data_scaled_oneclass) == -1
    
    # 對非異常樣本使用 SVM 進行分類
    predictions = []
    for i, is_unknown in enumerate(is_anomaly):
        if is_unknown:
            predictions.append('unknown')
        else:
            # 使用 SVM 預測
            single_sample = test_data_scaled_svm[[i]]
            svm_pred = svm_model.predict(single_sample)[0]
            class_name = label_encoder.inverse_transform([svm_pred])[0]
            predictions.append(class_name)
    
    return pd.Series(predictions), anomaly_scores