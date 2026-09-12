# -*- coding: utf-8 -*-
"""
Created on Sat May 31 16:17:07 2025

@author: HP
"""
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score

#輸出分類後的report結果(包括未知類+已之類 和 只有已知類的)
def printEvalaution(test_label_processed,y_pred_final):
    #Tranfer the labels into str
    test_label_processed = test_label_processed.astype(str)
    y_pred_final = pd.Series(y_pred_final).astype(str)
    #print out classify report including both known and unknown classes
    print("\n==== Result including unknown class ====")
    print(classification_report(test_label_processed,y_pred_final,zero_division=0))
    
    #print out the report only including the known classes
    Already_known=test_label_processed != 'unknown'

    if Already_known.sum() == 0:
        print("\n==== Result with only known class ====")
        print("No known classes found in the test labels.")
        return test_label_processed, y_pred_final

    print("\n==== Result with only known class ====")
    print(classification_report(test_label_processed[Already_known],y_pred_final[Already_known],zero_division=0))
    print("Accuracy on known classes: ",accuracy_score(test_label_processed[Already_known], y_pred_final[Already_known]))

    #Unknown classes count
    unknown_count=(test_label_processed=='unknown').sum()
    print(f"\nThere are {unknown_count} samples labeled as unknown class")
    print()
    
    return test_label_processed,y_pred_final