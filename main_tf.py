import os
import datetime

import joblib
import numpy as np
import tensorflow as tf
import MySQLdb as sql
from utils import load_data, plot_history_tf, plot_heat_map, recall_rate

import numpy as np
from scipy.interpolate import interp1d
from gcforest.gcforest import GCForest
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# project root path
project_path = "./"
# define log directory
# must be a subdirectory of the directory specified when starting the web application
# it is recommended to use the date time as the subdirectory name
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "ecg_model_CNN.h5"

# the ratio of the test set
RATIO = 0.3
# the random seed
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 30

start_time = 0
# build the CNN model
# 卷积神经网络构建
def build_model():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300,)),
        # reshape the tensor with shape (batch_size, 300) to (batch_size, 300, 1)
        tf.keras.layers.Reshape(target_shape=(300, 1)),
        # the first convolution layer, 4 21x1 convolution kernels, output shape (batch_size, 300, 4)
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='relu'),
        # the first pooling layer, max pooling, pooling size=3 , stride=2, output shape (batch_size, 150, 4)
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # the second convolution layer, 16 23x1 convolution kernels, output shape (batch_size, 150, 16)
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # the second pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 75, 16)
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # the third convolution layer, 32 25x1 convolution kernels, output shape (batch_size, 75, 32)
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='relu'),
        # the third pooling layer, average pooling, pooling size=3, stride=2, output shape (batch_size, 38, 32)
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # the fourth convolution layer, 64 27x1 convolution kernels, output shape (batch_size, 38, 64)
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
        # flatten layer, for the next fully connected layer, output shape (batch_size, 38*64)
        tf.keras.layers.Flatten(),
        # fully connected layer, 128 nodes, output shape (batch_size, 128)
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout layer, dropout rate = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # fully connected layer, 5 nodes (number of classes), output shape (batch_size, 5)
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return newModel

# gcForest模型构建、训练与准确率测试
def gcForest():
    start_time = time.time()
    def get_toy_config():
        config = {}
        ca_config = {}
        ca_config["random_state"] = 0
        ca_config["max_layers"] = 100 #最大的层数，layer对应论文中的level
        ca_config["early_stopping_rounds"] = 3
        ca_config["n_classes"] = 5  # 类别数
        ca_config["estimators"] = []
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
        print("训练用时：", train_end_time - start_time)

    y_predict = model.predict(np.array(X_test))  # 预测
    print(y_predict)
    print("准确率:", accuracy_score(y_test, y_predict))
    print("召回率：", recall_rate(y_test, y_predict))



# 模型训练与准确率测试
def CNN():
    # X_train,y_train is the training set
    # X_test,y_test is the test set
    X_train, X_test, y_train, y_test = load_data(RATIO, RANDOM_SEED)
    start_time = time.time()

    if os.path.exists(model_path):
        # import the pre-trained model if it exists
        print('Import the pre-trained model, skip the training process')
        model = tf.keras.models.load_model(filepath=model_path)
    else:
        # build the CNN model
        model = build_model()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        # define the TensorBoard callback object
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # train and evaluate model
        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            callbacks=[tensorboard_callback])
        # save the model
        model.save(filepath=model_path)
        # plot the training history
        plot_history_tf(history)
        train_end_time = time.time()
        print("模型训练用时：", train_end_time - start_time)

    # predict the class of test data
    # y_pred = model.predict_classes(X_test)  # predict_classes has been deprecated
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    # plot confusion matrix heat map
    plot_heat_map(y_test, y_pred)

    t = y_test == y_pred
    count = 0
    for i in t:
        if i :
            count = count + 1

    test_end_time = time.time()
    print("准确率为：", count/len(t))
    print("召回率：", recall_rate(y_test, y_pred))
    # print(X_test[0])

def CNN_predict():
    start_time = time.time()
    if os.path.exists(model_path):
        # import the pre-trained model if it exists
        print('Import the pre-trained model, skip the training process')
        model = tf.keras.models.load_model(filepath=model_path)

        # X_train, X_test, y_train, y_test = load_data(RATIO, RANDOM_SEED)
        #
        # test_arr = np.array([])
        # test_arr = np.expand_dims(X_test[0], axis=0)
        # print(np.shape(test_arr))
        # print(np.argmax(model.predict(test_arr)))

        arr32 = np.array(read_intelligent_mattress())

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
            print("N") # 正常心拍
        elif recognition_result == 1:
            print("A") # 房性早搏心拍(房性期前收缩)
        elif recognition_result == 2:
            print("V") # 室性期前收缩
        elif recognition_result == 3:
            print("L") # 左束支阻滞心拍
        elif recognition_result == 4:
            print("R")# 右束支阻滞心拍
    predict_end_time = time.time()
    print("预测用时：", predict_end_time - start_time)



def gcForest_predict():
    start_time = time.time()
    def get_toy_config():
        config = {}
        ca_config = {}
        ca_config["random_state"] = 0
        ca_config["max_layers"] = 100 #最大的层数，layer对应论文中的level
        ca_config["early_stopping_rounds"] = 3
        ca_config["n_classes"] = 5  # 类别数
        ca_config["estimators"] = []
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

        arr32 = np.array(read_intelligent_mattress())

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
    print("预测用时：", predict_end_time - start_time)



# 数据库配置
connect = sql.connect(
        host="localhost",  # mysql数据库的主机，mysql默认不允许root用户远程链接
        user="root",  # mysql服务器的用户名
        passwd="123456",  # mysql服务器用户的密码
        db="equipment",  # 数据库的名字
        # port = 3306 #端口号，默认为3306,可以不写
        # charest ="utf8" #链接数据库的字符集即编码
)
cursor = connect.cursor()

# 读取数据库数据
def read_intelligent_mattress():
    read_sql = "select * from intelligent_mattress order by id desc limit 1"
    cursor.execute(read_sql)
    rst = cursor.fetchone()
    # print(rst[11])  # 心率原始数据
    return eval(rst[11])


if __name__ == '__main__':
    # CNN()
    # gcForest()

    CNN_predict() # 用CNN预测运行速度更快一些
    # gcForest_predict()