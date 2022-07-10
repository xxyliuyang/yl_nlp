import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import roc_curve, precision_recall_curve
from collections import defaultdict

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def plot_roc_curve(fper, tper, optimal_th, optimal_point):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def train_test_split(data_X, cls_lab):
    label_map = defaultdict(list)
    for i, lable in enumerate(list(cls_lab)):
        label_map[lable].append(i)

    lable0 = label_map[0]
    lable1 = label_map[1]
    random.shuffle(lable0)
    random.shuffle(lable1)
    train_index = lable0[:100] + lable1[:100]
    test_index = lable0[-100:] + lable1[-10:]
    random.shuffle(train_index)
    random.shuffle(test_index)

    train_X, train_y = data_X[train_index], cls_lab[train_index]
    test_X, test_y = data_X[test_index], cls_lab[test_index]
    return train_X, train_y, test_X, test_y

def get_ROC_info():
    model = svm.SVC(kernel='linear', probability=True)
    model = model.fit(train_X, train_y)
    test_y_score = model.decision_function(test_X)
    prediction = model.predict(test_X)

    fper, tper, thresholds = roc_curve(test_y, test_y_score)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=fper, FPR=tper, threshold=thresholds)
    # precision, recall, thresholds = precision_recall_curve(test_y, test_y_score)
    return fper, tper, optimal_th, optimal_point


if __name__ == '__main__':
    data_X, cls_lab = make_classification(n_samples=10000, n_classes=2, weights=[1, 1], random_state=2)
    train_X, train_y, test_X, test_y = train_test_split(data_X, cls_lab)

    fper, tper, optimal_th, optimal_point = get_ROC_info()
    plot_roc_curve(fper, tper, optimal_th, optimal_point)

