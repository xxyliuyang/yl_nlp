import numpy as np
import copy
import pandas as pd
from sklearn.datasets import load_iris
import xgboost as xgb
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import accuracy_score

def func_get_data(N_data, N_feature):
    x = np.random.normal(loc = 0,
                         scale = 1.0,
                         size = (N_data, N_feature))
    y = 2 * x.T[0] + np.exp(x.T[3])
    return x, y

def func_permutation(model, x, y, err):
    for index_feature in range(len(x[0])):
        list_score = []
        for index_sample in range(len(x)):
            dat = copy.deepcopy(x[0])
            ans = y[0]
            for index_permute in range(1, len(x)):
                dat[index_feature] = x[index_permute][index_feature]
                pred = model.predict(dat.reshape(1, -1))
                score = accuracy_score([ans],
                                           pred)
                list_score.append(score)

            x = np.roll(x, 1, axis=0)
            y = np.roll(y, 1, axis=0)

        print("Permute feature:", index_feature, ", get importance:", np.mean(list_score) - err)

if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)  # 转化成DataFrame格式
    target = iris.target

    xgb_model = xgb.XGBClassifier()
    clf = xgb_model.fit(df.values, target)

    a = clf.feature_importances_
    xgb.plot_importance(xgb_model)
    features = pd.DataFrame(sorted(zip(a, df.columns), reverse=True))

    # 调用
    perm = PermutationImportance(xgb_model, random_state=1).fit(df, target)  # 实例化
    eli5.show_weights(perm)

    # 手动
    error = accuracy_score(target, clf.predict(df))
    func_permutation(xgb_model, df.values, target, error)
