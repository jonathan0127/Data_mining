import sys
from pathlib import Path
from preprocessing import data_read_and_preprocess
from RandomForest import RandomForestClassifierWrapper, train_oneclass_svm, predict_with_oneclass_svm, predict_with_threshold
import ClassifyEvaluate
import numpy as np
from DBSCAN import DBSCAN
from Kmeans import KMeansCluster
from Kmeans import find_best_k
import SVM
_BASE_DIR = Path(__file__).resolve().parent
for _sub in ["models", "clustering", "utils"]:
    _p = str(_BASE_DIR / _sub)
    if _p not in sys.path:
        sys.path.append(_p)
if str(_BASE_DIR) not in sys.path:
    sys.path.append(str(_BASE_DIR))



def clusterToLabel(cluster_labels, original_labels):
    map = {} # Dictionary
    for cluster in np.unique(cluster_labels):
        if cluster == -1:  continue
        mask = (cluster_labels == cluster)
        mode_label = original_labels[mask].mode()[0]
        map[cluster] = mode_label

    new_labels = [map.get(x) for x in cluster_labels]
    return new_labels


def main(oneclass, pca, standard, classify_algorithm, clustering_algorithm):
    """
    main oneclassm, pca, standard 三個參數可以調整
    都是 bool ，True 代表啟用 Flase 則是關閉
    回傳 accuracy, known_accuracy
    """

    # Preprocess the data
    preprocess_result = data_read_and_preprocess(pca, standard)
    train_data = preprocess_result["train_data"]
    train_label_encoded = preprocess_result["train_label_encoded"]
    test_data = preprocess_result["test_data"]
    test_label_processed = preprocess_result["test_label_processed"]
    test_label_fulled = preprocess_result["test_label_fulled"]
    le = preprocess_result["label_encoder"]

    # Classification
    classify_model = None
    if classify_algorithm == 1:  # SVM
        classify_model, _ = SVM.train_svm(train_data, train_label_encoded)
    elif classify_algorithm == 2:  # Random Forest
        rf_wrapper = RandomForestClassifierWrapper()
        classify_model = rf_wrapper.train(train_data, train_label_encoded)

    if classify_model is None:
        raise RuntimeError("Failed to initialize classify_model. Please check the classification algorithm logic.")

    # 選擇預測方法：True 使用 One-Class SVM，False 使用 threshold 方法
    if oneclass:
        svm_model, svm_scaler = train_oneclass_svm(train_data)
        y_pred_final, anomaly_scores = predict_with_oneclass_svm(classify_model, svm_model, svm_scaler, test_data, le)
        print("Using One-Class SVM for anomaly detection")
    else:
        y_pred_final, max_probs, best_threshold = predict_with_threshold(classify_model, test_data, le, test_label_processed)
        print("Using confidence threshold method for unknown detection")

    ClassifyEvaluate.printEvalaution(y_pred_final, test_label_processed)

    unknown_mask = y_pred_final == 'unknown'
    unknown_indices = np.where(unknown_mask)[0]
    unknown_data = test_data[unknown_mask]
    
    if len(unknown_indices) > 0:
        unknown_data_np = unknown_data.values if hasattr(unknown_data, 'values') else unknown_data
        
        cluster_labels = None
        if clustering_algorithm == 1: # Kmeans
            best_k = find_best_k(unknown_data_np, cutoff=1e-3, max_iter=100, max_k=10)
            kmeans = KMeansCluster(best_k)
            cluster_labels = kmeans.fit(unknown_data_np)
        elif clustering_algorithm == 2: # DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(unknown_data_np)

        print("Cluster labels for unknown data:", cluster_labels)
    
        # 將標籤映射到原始 index
        unknown_labels = clusterToLabel(cluster_labels, test_label_fulled[unknown_mask])

        # 將未知樣本的標籤更新到 y_pred_final 中
        y_pred_final.loc[unknown_indices] = unknown_labels

        unknown_correct = (y_pred_final[unknown_mask] == test_label_fulled[unknown_mask]).sum()
        known_correct = (y_pred_final[~unknown_mask] == test_label_fulled[~unknown_mask]).sum()
        total_correct = unknown_correct + known_correct
        accuracy = total_correct / len(test_label_fulled)
        # print("Total number of samples:", len(test_label_fulled))
        # print("known samples:", len(test_label_fulled[~unknown_mask]))
        # print("known correct:", known_correct)
        # print("unknown samples:", len(test_label_fulled[unknown_mask]))
        # print("unknown correct:", unknown_correct)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy, (known_correct / len(test_label_fulled[~unknown_mask]))

if __name__ == "__main__":

    # 選擇要測試的演算法
    print("選擇要測試的分類演算法：")
    print("1. SVM")
    print("2. Random Forest")
    classify_algorithm = int(input("請輸入選擇 (1 或 2)："))
    if classify_algorithm > 2: # 如果輸入不在範圍內，則預設使用 Random Forest
        classify_algorithm = 2
    print("選擇要測試的分群演算法：")
    print("1. KMeans")
    print("2. DBSCAN")
    clustering_algorithm = int(input("請輸入選擇 (1 或 2)："))
    if clustering_algorithm > 2: # 如果輸入不在範圍內，則預設使用 KMeans
        clustering_algorithm = 1

    # oneclass, PCA, standard
    tupels = [
        (False, False, False),
        (False, False, True),
        (False, True, True),
        (True, False, True),
        (True, True, True),
    ]

    ans = []
    for i in range(len(tupels)):
        oneclass, pca, standard = tupels[i]
        print("param:", oneclass, pca, standard)
        accuracy, known_accuracy = main(oneclass, pca, standard, classify_algorithm, clustering_algorithm)
        ans.append([accuracy, known_accuracy])

    for i in range(len(ans)):
        print(f"acc: {ans[i][0]:.4f}, kacc: {ans[i][1]:.4f}")
