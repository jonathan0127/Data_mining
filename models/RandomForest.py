import sys
from pathlib import Path

# Add project root and subdirectories to sys.path
_ROOT = Path(__file__).resolve().parent.parent
for _d in [_ROOT, _ROOT / "models", _ROOT / "clustering", _ROOT / "utils"]:
    if str(_d) not in sys.path:
        sys.path.append(str(_d))

import pandas as pd
import numpy as np
from preprocessing import data_read_and_preprocess
from HandleUnknown import Best_Predict_Confidence_Threshold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
import ClassifyEvaluate  
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from Kmeans import kmeans, find_best_k
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from itertools import permutations
import numpy as np
from sklearn.metrics import accuracy_score
import umap.umap_ as umap
import warnings
from params import use_oneclass_svm
from preprocessing import predict_with_oneclass_svm
from scipy.stats import ttest_ind,f_oneway
warnings.filterwarnings('ignore')

class RandomForestClassifierWrapper:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.best_params_ = None

    def train(self, train_data, train_label):
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': [None, 'balanced']
        }
        grid_search = GridSearchCV(model, param_grid, scoring='f1_macro', cv=5, n_jobs=-1, verbose=0)
        grid_search.fit(train_data, train_label)
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        print("Best parameter combination:", self.best_params_)
        return self.model

    def predict(self, test_data):
        return self.model.predict(test_data)

    def predict_proba(self, test_data):
        return self.model.predict_proba(test_data)

#A Function to do Randomforest training

#用 One-class SVM 訓練調整模型
def train_oneclass_svm(train_data):
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    oneclass_svm = OneClassSVM(kernel='rbf',gamma='scale',nu=0.15)
    oneclass_svm.fit(train_data_scaled)
    return oneclass_svm, scaler

def predict_with_threshold(model, test_data, label_encoder, true_labels, threshold_func=Best_Predict_Confidence_Threshold):
    #找出每個樣本屬於每一類別個別的機率
    prob_pred = model.predict_proba(test_data)
    #機率最高的那一類
    max_probs = np.max(prob_pred, axis=1)
    class_pred = model.predict(test_data)
    class_pred_labels = label_encoder.inverse_transform(class_pred)

    #最終預測結果
    y_pred_final = np.where(max_probs < 0.75, 'unknown', class_pred_labels)
    return pd.Series(y_pred_final), max_probs, class_pred_labels

def cluster_unknown_samples(test_data, y_pred_final):
    test_data=test_data.reset_index(drop=True)
    y_pred_final = pd.Series(y_pred_final).reset_index(drop=True)
    unknown_mask = (y_pred_final == 'unknown')
    unknown_data = test_data[unknown_mask]
    if len(unknown_data) < 2:
        print("Not enough unknown samples to cluster.")
        return None
    print(f"Total unknown samples: {len(unknown_data)}")
    X_unknown = unknown_data.copy()

    scaler = StandardScaler()
    unknown_data = scaler.fit_transform(unknown_data)
    #找出最佳的k值
    best_k = find_best_k(unknown_data, cutoff=1e-3, max_iter=100)
    print(f"[Unknown Clustering] Best k for unknown samples: {best_k}")
    cluster_labels, _ = kmeans(unknown_data, best_k, cutoff=1e-3, max_iter=100)
    
    print("\n[Statistical Testing Across Clusters]")
    
    X_unknown_np = np.array(X_unknown)
    unique_clusters = np.unique(cluster_labels)
      
    if len(unique_clusters) <= 1:
        print("Cannot perform T-test or ANOVA: Only one cluster.")
    elif len(unique_clusters) == 2:  #2群做T檢定
        idx1 = cluster_labels == unique_clusters[0]
        idx2 = cluster_labels == unique_clusters[1]
        for feature_index in range(X_unknown_np.shape[1]):
            t_stat, p_val = ttest_ind(X_unknown_np[idx1, feature_index], X_unknown_np[idx2, feature_index], equal_var=False)
            print(f"Feature {feature_index:3d}: t = {t_stat:.4f}, p = {p_val:.4e}")
    else:   #3群以上做ANOVA檢定
        print("\n[ANOVA per feature across clusters]")
        for feature_index in range(X_unknown_np.shape[1]):
            groups = [X_unknown_np[cluster_labels == cluster, feature_index] for cluster in unique_clusters]
            group_sizes = [len(g) for g in groups]
        #排除樣本數少的
        if all(size > 1 for size in group_sizes):
            try:
                f_stat, p_val = f_oneway(*groups)
                print(f"Feature {feature_index}: F = {f_stat:.4f}, p = {p_val:.4e}")
            except Exception as e:
                print(f"Feature {feature_index}: Error running ANOVA: {e}")
            else:
                print(f"Feature {feature_index} skipped due to insufficient data in groups: {group_sizes}")    
    #UMAP畫圖
    n_samples = len(unknown_data)
    if n_samples < 5:
        print("No enough samples for UMAP")
    try:
        n_neighbors = min(15, n_samples - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        unknown_embedded = reducer.fit_transform(unknown_data)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=unknown_embedded[:, 0], y=unknown_embedded[:, 1], hue=cluster_labels, palette='viridis')
        plt.title("UMAP projection of unknown class clusters")
        plt.legend(title="Cluster")
        plt.show()
    except ValueError as error:
        print(f"UMAP error: {error}")
    
    pca = PCA(n_components=2)
    unknown_pca = pca.fit_transform(unknown_data)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=unknown_pca[:, 0], y=unknown_pca[:, 1], hue=cluster_labels, palette='viridis')
    plt.title("PCA projection of unknown class clusters")
    plt.legend(title="Cluster")
    plt.show()

    return cluster_labels, unknown_embedded
    
def cluster_known_subgroups(test_data, y_pred_final, label_encoder):
    #排除unknown類別
    test_data=test_data.reset_index(drop=True)
    y_pred_final=pd.Series(y_pred_final).reset_index(drop=True)
    known_mask = y_pred_final != 'unknown'
    known_data = test_data[known_mask]
    known_labels = y_pred_final[known_mask]
    #標準化
    scaler = StandardScaler()
    known_data_scaled = scaler.fit_transform(known_data)
    results = []
    for class_name in np.unique(known_labels):
        cls_mask = (known_labels == class_name)
        cls_data = known_data_scaled[cls_mask]
        if len(cls_data) < 3:
            print(f"[Known Subgroups] Class {class_name} too small to cluster.")
            continue

        best_k = find_best_k(cls_data, cutoff=1e-3, max_iter=100)
        print(f"[Known Subgroups] Class {class_name} best k: {best_k}")
        cluster_labels, _ = kmeans(cls_data, best_k, cutoff=1e-3, max_iter=100)
        # 統計每子群特徵平均，排除NaN
        cluster_info = []
        for cluster_id in np.unique(cluster_labels):
            indices = np.where(cluster_labels == cluster_id)[0]
            subgroup_size = len(indices)
            subgroup_mean = np.nanmean(cls_data[indices], axis=0)  # 用nanmean避免nan干擾

            cluster_info.append({
                "class": class_name,
                "cluster_id": cluster_id,
                "size": subgroup_size,
                "mean_features": subgroup_mean
            })

        results.append(cluster_info)
        # UMAP可視化
        n_samples = cls_data.shape[0]
        if n_samples < 5:
            print("No enough samples for UMAP")
            continue
        try:
        # 調整 n_neighbors 不超過 n_samples - 1
            n_neighbors = min(15, n_samples - 1)
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
            unknown_embedded = reducer.fit_transform(cls_data)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=unknown_embedded[:, 0], y=unknown_embedded[:, 1], hue=cluster_labels, palette='viridis')
            plt.title("UMAP projection of unknown class clusters")
            plt.legend(title="Cluster")
            plt.show()
        except ValueError as error:
            print(f"UMAP error: {error}")
        
        # PCA 可視化
        pca = PCA(n_components=2)
        cls_data_pca = pca.fit_transform(cls_data)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=cls_data_pca[:, 0], y=cls_data_pca[:, 1], hue=cluster_labels, palette='tab10')
        plt.title(f"PCA projection of {class_name} subgroups")
        plt.legend(title="Subgroup")
        plt.show()

    return results

def evaluate_kmeans_accuracy_given_preds(cluster_preds, true_labels):


    #取得所有出現過的「真實標籤」和「分群結果」的 cluster id
    true_label_set = sorted(set(true_labels))
    cluster_label_set = sorted(set(cluster_preds))

    #確保真實的類別數量與 cluster 數相同
    assert len(true_label_set) == len(cluster_label_set), "Mismatch in number of clusters and unique true labels!"

    best_accuracy = 0
    best_mapping = {}
    best_pred_labels = None
    
    #枚舉所有可能的對應方式
    for perm in permutations(true_label_set):
        mapping = dict(zip(cluster_label_set, perm))
        pred_labels_mapped = np.array([mapping[cluster] for cluster in cluster_preds])
        acc = accuracy_score(true_labels, pred_labels_mapped)
        if acc > best_accuracy:
            best_accuracy = acc
            best_mapping = mapping
            best_pred_labels = pred_labels_mapped

    return best_accuracy, best_mapping, best_pred_labels

def evaluate_unknown_with_kmeans(X_test, y_test, y_pred_final, k=7):
    unknown_mask = (y_pred_final == 'unknown')
    
    # 確保 X_test 是 DataFrame 或 ndarray
    if hasattr(X_test, 'values'):
        X_unknown = X_test.loc[unknown_mask].values
    else:
        X_unknown = X_test[unknown_mask]

    # 確保 y_test 是 Series 或 ndarray
    if hasattr(y_test, 'loc'):
        y_unknown = y_test.loc[unknown_mask]
    else:
        y_unknown = y_test[unknown_mask]

    # 替換 'unknown' 字串為 -1
    y_unknown = y_unknown.replace('unknown', -1)

    # 再轉成 int
    y_unknown = y_unknown.astype(int).values

    cluster_preds, centroids = kmeans(X_unknown, k=k, cutoff=0.01, max_iter=100)
    acc, mapping, predicted_labels = evaluate_kmeans_accuracy_given_preds(cluster_preds, y_unknown)

    print(f"\n[Unknown KMeans Accuracy Evaluation using kmeans]")
    print(f"Best Accuracy: {acc:.4f}")
    print(f"Best Cluster to Label Mapping: {mapping}")
    return acc, mapping, predicted_labels


def main():
    preprocess_result = data_read_and_preprocess()
    train_data = preprocess_result["train_data"]
    train_label_encoded = preprocess_result["train_label_encoded"]
    test_data = preprocess_result["test_data"]
    test_label_processed = preprocess_result["test_label_processed"]
    test_label_fulled = preprocess_result["test_label_fulled"]
    le = preprocess_result["label_encoder"]

    rf_wrapper = RandomForestClassifierWrapper()
    rf_model = rf_wrapper.train(train_data, train_label_encoded)

    # 選擇預測方法：True 使用 One-Class SVM，False 使用 threshold 方法
    if use_oneclass_svm:
        svm_model, svm_scaler = train_oneclass_svm(train_data)
        y_pred_final, anomaly_scores = predict_with_oneclass_svm(rf_model, svm_model, svm_scaler, test_data, le)
        print("Using One-Class SVM for anomaly detection")
    else:
        y_pred_final, max_probs = predict_with_threshold(rf_model, test_data, le, test_label_processed)
        print("Using confidence threshold method for unknown detection")

    print(y_pred_final)
    print(y_pred_final.value_counts())
    print(test_label_processed.value_counts())
    ClassifyEvaluate.printEvalaution(y_pred_final, test_label_processed)
    
    # 計算並輸出準確率
    accuracy = (y_pred_final == test_label_processed).mean()
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Number of samples classified as unknown: {(y_pred_final == 'unknown').sum()}")

    #Clustering: Kmeans
    # 針對 unknown 類別聚類
    cluster_unknown_samples(test_data, y_pred_final)

    # # 針對 known 類別分群
    cluster_known_subgroups(test_data, y_pred_final, le)

    #Evalaution
    evaluate_unknown_with_kmeans(test_data, test_label_processed, y_pred_final)



if __name__ == "__main__":
    main()
