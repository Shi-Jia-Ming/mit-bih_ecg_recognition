import wfdb
import pywt
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# wavelet denoise preprocess using mallat algorithm
def denoise(data, number):
    # wavelet decomposition
    # 进行9尺度小波变换，分解信号，发现D1和D2层是高频噪声的主要集中区域
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # denoise using soft threshold
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # get the denoised signal by inverse wavelet transform
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')

    # 查看图形
    # plt.figure(figsize=(20, 4))
    # plt.plot(data[:2000])
    # plt.savefig(number + "_old_pic.png")
    # plt.show()
    #
    # plt.figure(figsize=(20, 4))
    # plt.plot(rdata[:2000])
    # plt.savefig(number + "_new_pic.png")
    # plt.show()

    return rdata


# load the ecg data and the corresponding labels, then denoise the data using wavelet transform
def get_data_set(number, X_data, Y_data):
    # 只选择正常心拍、房性早搏心拍(房性期前收缩)、室性期前收缩、左束支阻滞心拍、右束支阻滞心拍
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']

    # load the ecg data record
    print("loading the ecg data of No." + number)
    record = wfdb.rdrecord('ecg_data/' + number, channel_names=['MLII']) # 获取到一个650000*1的二维数组
    # print("获取的record长度：", np.shape(record.p_signal))
    data = record.p_signal.flatten()    # 重塑成650000长的一维数组
    print("data形状：", np.shape(data))
    rdata = denoise(data=data, number=number)

    # 获取R波的位置以及对应的标签
    annotation = wfdb.rdann('ecg_data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol
    # print("Rclass:",Rclass)

    # remove the unstable data at the beginning and the end
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # the data with specific labels (N/A/V/L/R) required in this record are selected, and the others are discarded
    # X_data: data points of length 300 around the R-wave
    # Y_data: convert N/A/V/L/R to 0/1/2/3/4 in order
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


# load dataset and preprocess
def load_data(ratio, random_seed):
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        get_data_set(n, dataSet, lableSet)

    # reshape the data and split the dataset
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(lableSet).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(dataSet, lableSet, test_size=ratio, random_state=random_seed)
    return X_train, X_test, y_train, y_test


# confusion matrix
def plot_heat_map(y_test, y_pred):
    con_mat = confusion_matrix(y_test, y_pred)
    # normalize
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # plot
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


def plot_history_tf(history):
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()


def plot_history_torch(history):
    plt.figure(figsize=(8, 8))
    plt.plot(history['train_acc'])
    plt.plot(history['test_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history['train_loss'])
    plt.plot(history['test_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()

# def get_data_from_MySQL():

def recall_rate(y_test, y_predict):
    test = 0
    predict = 0
    for i in range(len(y_test)):
        if y_test[i] != 0:
            test += 1
            if y_predict[i] == y_test[i]:
                predict += 1

    rate = predict / test
    return rate