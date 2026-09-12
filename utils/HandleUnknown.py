# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:03:23 2025

@author: HP
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

#找出最佳結果的閾值，當該樣本max_probs小於這值代表其預測出的類別結果不確定，將其歸為未知類樣本
def Best_Predict_Confidence_Threshold(max_probs,class_pred_labels,test_label_processed):
    #If the confidence for certain prediction is lower than threshold, classify it as unknown
    threshold=np.arange(0.45,0.95,0.01)
    accuracies = []
    #Trasfer all the processed test labels to str
    test_label_processed=test_label_processed.astype(str)
    for p in threshold:
        y_pred_threshold=np.where(max_probs<p, 'unknown', class_pred_labels)
        y_pred_threshold=pd.Series(y_pred_threshold).astype(str)
        accuracy = accuracy_score(test_label_processed, y_pred_threshold)
        accuracies.append(accuracy)
    best_threshold=threshold[np.argmax(accuracies)]
    return best_threshold