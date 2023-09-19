import os
import time
import datetime

import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d

from DbInterface.DbInterface import DbInterface
from Utils.utils import load_data, plot_history_tf, plot_heat_map, recall_rate


class CnnModule:
    # 数据集中训练集和测试集的比例
    RATIO = 0.3
    # 随机种子
    RANDOM_SEED = 42
    BATCH_SIZE = 128
    NUM_EPOCHS = 30

    # CNN模型文件的路径
    model_path = "./Module/ecg_model_CNN.h5"
    # 日志文件目录
    log_dir = "../logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def __init__(self):
        if os.path.exists(self.model_path):
            # import the pre-trained model if it exists
            # print('Import the pre-trained model, skip the training process')
            print('导入现存的模型，跳过模型的训练')
            self.model = tf.keras.models.load_model(filepath=self.model_path)
        else:
            # X_train,y_train is the training set
            # X_test,y_test is the test set
            X_train, X_test, y_train, y_test = load_data(self.RATIO, self.RANDOM_SEED)
            train_start_time = time.time()

            # build the CNN model
            self.model = self.__build_model()
            self.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
            self.model.summary()
            # define the TensorBoard callback object
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
            # train and evaluate model
            history = self.model.fit(X_train, y_train, epochs=self.NUM_EPOCHS,
                                     batch_size=self.BATCH_SIZE,
                                     validation_data=(X_test, y_test),
                                     callbacks=[tensorboard_callback])
            # save the model
            self.model.save(filepath=self.model_path)
            # plot the training history
            plot_history_tf(history)
            train_end_time = time.time()
            print("模型训练用时：", train_end_time - train_start_time)

            # predict the class of test data
            # y_pred = model.predict_classes(X_test)  # predict_classes has been deprecated
            y_pred = np.argmax(self.model.predict(X_test), axis=-1)
            # plot confusion matrix heat map
            plot_heat_map(y_test, y_pred)

            # t = y_test == y_pred
            t = np.array(y_test == y_pred)
            count = 0
            for i in t:
                if i:
                    count = count + 1

            test_end_time = time.time()
            print("测试模型用时：", test_end_time - train_end_time)
            print("准确率为：", count / len(t))
            print("召回率：", recall_rate(y_test, y_pred))
            # print(X_test[0])

    # 使用模型预测数据
    def CNN_predict(self):
        predict_start_time = time.time()
        # 如果已经训练好模型，则直接使用模型
        self.model = tf.keras.models.load_model(filepath=self.model_path)

        # X_train, X_test, y_train, y_test = load_data(RATIO, RANDOM_SEED)
        #
        # test_arr = np.array([])
        # test_arr = np.expand_dims(X_test[0], axis=0)
        # print(np.shape(test_arr))
        # print(np.argmax(model.predict(test_arr)))

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
        recognition_result = np.argmax(self.model.predict(arr300))

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

    def CNN_test(self):
        test_start_time = time.time()
        # X_train,y_train is the training set
        # X_test,y_test is the test set
        # 获取测试数据集
        X_train, X_test, y_train, y_test = load_data(self.RATIO, self.RANDOM_SEED)

        # 获取预测结果
        y_pred = np.argmax(self.model.predict(X_test), axis=-1)

        # 测试模型拟合度
        # t = y_test == y_pred
        t = np.array(y_test == y_pred)
        count = 0
        for i in t:
            if i:
                count = count + 1

        test_end_time = time.time()
        print("测试模型用时：", test_end_time - test_start_time)
        print("准确率为：", count / len(t))
        print("召回率：", recall_rate(y_test, y_pred))
        # print(X_test[0])

    # build the CNN model
    # 卷积神经网络构建
    @staticmethod
    def __build_model():
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
