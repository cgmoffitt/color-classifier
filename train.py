#In this file we will load the data and train the model
import keras
import os
import numpy as np
import tensorflow as tf
from keras import backend as K

model = 0


def prepare_data():
    colors = []
    labels = [
        "red",
        "green",
        "blue",
        "orange",
        "yellow",
        "pink",
        "purple",
        "brown",
        "grey"
    ]
    labelsList = []

    filepath = "clean_data.txt"
    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()
    with open(filepath) as fp:
        i = 0
        r = 0
        g = 0
        b = 0
        for line in fp:
            word = line.split(",")
            col = [float(word[0]) / 255, float(word[1]) / 255, float(word[2]) / 255]
            colors.append(col)
            labelsList.append(labels.index(word[3].rstrip("\n")))
    xs = np.array(colors)
    labelsList = np.array(labelsList)
    ys = keras.backend.one_hot(labelsList, 9)
    return xs, ys

def buildModel():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=16, activation='sigmoid', input_shape=[3]))
    model.add(keras.layers.Dense(units=9, activation='softmax'))
    opt = keras.optimizers.SGD(learning_rate=0.2)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model


def main():
    xs, ys = prepare_data()
    model = buildModel()
    model.fit(x=xs,y=ys, epochs=500) #train model


main()
