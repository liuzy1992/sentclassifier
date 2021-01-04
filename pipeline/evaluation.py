#!/usr/bin/env python3

from sklearn.metrics import classification_report, log_loss

def evaluation(df_test, clf):
    y_true, y_pred = df_test['label'], clf.predict(df_test['senttext'])
    
    # (precision, recall, f1, _) = precision_recall_fscore_support(y_true, y_pred, average='weighted', warn_for=tuple())
    # print("Precision: {:.6f}. Recall: {:.6f}. F1: {:.6f}.".format(precision, recall, f1))
    print("Classification report on test set: ")
    print(classification_report(y_true, y_pred, digits=4))
    # print("Loss: {:4f}".format(log_loss(y_true, y_pred)))
    # print(confusion_matrix(y_true, y_pred))
    # print("Accuracy score: {:.6f}".format(accuracy_score(y_true, y_pred).item()))
