#In this file we will load the data and train the model
import keras
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt

labels = [
    "red",
    "green",
    "blue",
    "orange",
    "yellow",
    "pink",
    "purple",
    "brown",
    "grey",
    "black"
]

#Hyper Parameters
NUM_EPOCHS = 500
LEARNING_RATE = 0.2
NUM_HIDDEN_UNITS = 16
NUM_TRAIN = 5500
VALIDATION_SPLIT = 0.1


def get_data_from_file():
    data = []
    filepath = "clean_data.txt"
    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()
    with open(filepath) as fp:
        entry = []
        for line in fp:
            word = line.split(",")
            r = word[0]
            g = word[1]
            b = word[2]
            label = word[3].rstrip("\n")
            entry = [r, g, b, label]
            data.append(entry)
    return data

def prepare_data(data):
    np.random.shuffle(data)
    data = np.array(data)

    #seperate input values
    xs = data[:,0:3]
    xs = np.divide(xs.astype(np.float), 255) #normalize

    #seperate label values
    ys = data[:, 3]
    ys = np.vectorize(labels.index)(ys)

    #seperate training data
    XTrain = xs[0:NUM_TRAIN,:]
    YTrain = ys[0:NUM_TRAIN]
    YTrain = keras.backend.one_hot(YTrain, 10)

    #sepearate testing data
    XTest = xs[NUM_TRAIN:, :]
    YTest = ys[NUM_TRAIN:]
    YTest = keras.backend.one_hot(YTest, 10)

    return XTrain, YTrain, XTest, YTest

def buildModel():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=NUM_HIDDEN_UNITS, activation='sigmoid', input_shape=[3]))
    model.add(keras.layers.Dense(units=10, activation='softmax'))
    opt = keras.optimizers.SGD(learning_rate=0.2)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    return model

def saveModelAsTFLite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.post_training_quantize = True
    tflite_buffer = converter.convert()
    open( 'android/model.tflite' , 'wb' ).write( tflite_buffer )

def plotTrainingHistory(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def calculateF1(precision, recall):
    return 2 * ((precision * recall)/ (precision + recall))

def displayResults(results):
    loss = results[0]
    accuracy = results[1]
    precision = results[2]
    recall = results[3]
    f1_score = calculateF1(precision, recall)
    print('\n\nResults from evaluating on training data: \n')
    print('Loss', round(loss, 2))
    print('Accuracy', round(accuracy, 2))
    print('Precision: ', round(precision, 2))
    print('Recall: ', round(recall, 2))
    print('F1 Score: ', round(f1_score, 2))
    print('\n\n')


def main():
    data = get_data_from_file()
    XTrain, YTrain, XTest, YTest = prepare_data(data)
    model = buildModel()
    history = model.fit(x=XTrain,y=YTrain, epochs=NUM_EPOCHS, validation_split=VALIDATION_SPLIT) #train model
    plotTrainingHistory(history)
    displayResults(model.evaluate(x=XTrain, y=YTrain))
    displayResults(model.evaluate(x=XTest, y=YTest))


main()
