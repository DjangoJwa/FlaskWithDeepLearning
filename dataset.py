# CIFAR-10 Small Image Classification
from keras.datasets import cifar10

def loadCIFAR10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)
