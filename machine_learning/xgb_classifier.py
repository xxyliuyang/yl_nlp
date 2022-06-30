"""用房价预测训练xgboost"""
from  sklearn import datasets
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def load_data():
    """房价数据"""
    boston = datasets.load_boston()
    data = pd.DataFrame(boston.data)
    data.columns = boston.feature_names
    data['price'] = boston.target
    y = data.pop('price')
    return data, y

def xgb_model():
    xg_reg = xgb.XGBRegressor(
        objective='reg:linear',
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=10,
        alpha=10
    )
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
    xg_reg.fit(x_train, y_train)
    pred = xg_reg.predict(x_test)
    mse = mean_squared_error(pred, y_test)
    print("mse：", mse)

def xgb_cv():
    params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
              'max_depth': 5, 'alpha': 10}
    cv_results = xgb.cv(dtrain=data_matrix, params=params, nfold=3,
                        num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
    cv_results.head()
    print((cv_results["test-rmse-mean"]).tail(1))

def xgb_show():
    params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
              'max_depth': 5, 'alpha': 10}
    xg_reg = xgb.train(params=params, dtrain=data_matrix, num_boost_round=10)
    import matplotlib.pyplot as plt

    # xgb.plot_tree(xg_reg, num_trees=0)
    # plt.rcParams['figure.figsize'] = [80, 60]
    # plt.show()

    xgb.plot_importance(xg_reg)
    plt.rcParams['figure.figsize'] = [3, 3]
    plt.show()

def xgb_grid():
    cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
    params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    model = xgb.XGBRegressor(**params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(x_train, y_train)


if __name__ == '__main__':
    data, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
    data_matrix = xgb.DMatrix(data, y)

    xgb_model()
    xgb_cv()
    xgb_show()
    xgb_grid()