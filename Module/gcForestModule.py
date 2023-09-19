import os
import time
import joblib

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score
from gcforest.gcforest import GCForest

from DbInterface.DbInterface import DbInterface
from Utils.utils import load_data, recall_rate

# 数据集中训练集和测试集的比例
RATIO = 0.3
# 随机种子
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 30


# gcForest模型构建、训练与准确率测试
def gcForest():
    train_start_time = time.time()

    def get_toy_config():
        config = {}
        ca_config = {"random_state": 0, "max_layers": 100, "early_stopping_rounds": 3, "n_classes": 5, "estimators": []}

        ca_config["estimators"].append(
            {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
             "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1})
        ca_config["estimators"].append(
            {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
        ca_config["estimators"].append(
            {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
        ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
        config["cascade"] = ca_config
        return config

    # Press the green button in the gutter to run the script.

    X_train, X_test, y_train, y_test = load_data(RATIO, RANDOM_SEED)
    print("GCForest model ...")
    if os.path.exists("gcForestModel.sav"):
        # 模型加载
        model = joblib.load('gcForestModel.sav')
    else:
        model = GCForest(get_toy_config())  # 构建模型
        model.fit_transform(X_train, y_train)  # 训练
        # 模型保存
        joblib.dump(model, 'gcForestModel.sav')
        train_end_time = time.time()
        print("训练用时：", train_end_time - train_start_time)

    y_predict = model.predict(np.array(X_test))  # 预测
    print(y_predict)
    print("准确率:", accuracy_score(y_test, y_predict))
    print("召回率：", recall_rate(y_test, y_predict))


# gcForest 模型预测数据
def gcForest_predict():
    predict_start_time = time.time()

    def get_toy_config():
        config = {}
        ca_config = {"random_state": 0, "max_layers": 100, "early_stopping_rounds": 3, "n_classes": 5, "estimators": []}

        ca_config["estimators"].append(
            {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
             "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1})
        ca_config["estimators"].append(
            {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
        ca_config["estimators"].append(
            {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
        ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
        config["cascade"] = ca_config
        return config

    # Press the green button in the gutter to run the script.
    if os.path.exists("gcForestModel.sav"):
        # 模型加载
        model = joblib.load('gcForestModel.sav')

        dbConnect = DbInterface("localhost", "root", "password", "equipment")
        arr32 = np.array(dbConnect.read_intelligent_mattress())

        # 生成长度为300的索引数组
        x = np.linspace(0, 31, num=32)
        x_new = np.linspace(0, 31, num=300)

        # 用线性插值将长度从32扩展到300
        f = interp1d(x, arr32, kind='linear')
        arr300 = f(x_new)

        # print(arr300.shape)  # 输出(300,)
        # print(arr300)
        arr300 = np.expand_dims(arr300, axis=0)
        recognition_result = np.argmax(model.predict(arr300))

        if recognition_result == 0:
            print("N")  # 正常心拍
        elif recognition_result == 1:
            print("A")  # 房性早搏心拍(房性期前收缩)
        elif recognition_result == 2:
            print("V")  # 室性期前收缩
        elif recognition_result == 3:
            print("L")  # 左束支阻滞心拍
        elif recognition_result == 4:
            print("R")  # 右束支阻滞心拍
    predict_end_time = time.time()
    print("预测用时：", predict_end_time - predict_start_time)
